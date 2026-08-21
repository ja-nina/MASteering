"""vLLM-based rollout engine for GRPO persona LoRA training.

Key speedup: instead of K sequential model.generate() calls, a single
vLLM call with SamplingParams(n=K) produces all K candidates in parallel
using PagedAttention + continuous batching.

GPU layout (typical 2-GPU node):
  cuda:0 — vLLM engine  (generation, memory-heavy)
  cuda:1 — HF PeftModel (probe_text, recompute_logprobs, grpo_step)

Weight sync after each optimizer step:
  1. peft_model.save_pretrained("/dev/shm/lora_sync/")   ~50 ms
  2. vllm_engine.load_lora("/dev/shm/lora_sync/")        ~100 ms
  Total overhead per step: ~150 ms vs minutes of generation saved.

Limitation: vLLM loads ONE LoRA adapter at a time.  We load adapter_a
(the SVD-constrained persona LoRA).  adapter_b is still trained on GPU 1
using the same episode data, but generation runs under adapter_a only.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch

from testbed.training.rollout import Episode, TurnGroup, TurnRecord, TRAINEE_ID, OPPONENT_ID


class VLLMRolloutEngine:
    """vLLM engine for fast multi-candidate generation.

    All generation (trainee + opponent) goes through this engine:
      - Trainee: ONE call with SamplingParams(n=K) → K candidates
      - Opponent: ONE call with no LoRA → base-model response
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda:0",
        max_lora_rank: int = 32,
        gpu_memory_utilization: float = 0.90,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 300,
    ):
        from vllm import LLM

        self.model_id   = model_id
        self.temperature = temperature
        self.top_p      = top_p
        self.top_k      = top_k
        self.max_tokens = max_tokens

        print(f"  [vLLM] loading {model_id} "
              f"(gpu_mem={gpu_memory_utilization})...", flush=True)
        self.llm = LLM(
            model=model_id,
            enable_lora=True,
            max_loras=2,                         # current + one being swapped in
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self.tokenizer   = self.llm.get_tokenizer()
        self._lora_id    = 0          # monotonically increasing; identifies adapter
        self._lora_req   = None       # current LoRARequest (None until load_lora)
        print("  [vLLM] ready.", flush=True)

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    def load_lora(self, lora_path: str) -> None:
        """Load (or hot-swap) LoRA adapter from a directory.

        Assigns a fresh id so vLLM doesn't serve a stale cached copy.
        The previous adapter is removed after the new request is installed.
        """
        from vllm.lora.request import LoRARequest

        old_id = self._lora_id
        self._lora_id += 1
        self._lora_req = LoRARequest("trainee", self._lora_id, lora_path)

        if old_id > 0:
            try:
                self.llm.llm_engine.remove_lora(old_id)
            except Exception:
                pass   # harmless if adapter was never cached

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _sampling_params(self, n: int):
        from vllm import SamplingParams
        return SamplingParams(
            n=n,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        )

    def generate_candidates(
        self,
        system_prompt: str,
        user_prompt: str,
        K: int,
    ) -> List[Tuple[str, torch.Tensor, int]]:
        """Generate K candidates from one prompt in a SINGLE vLLM call.

        Returns list of K (text, full_ids_cpu, input_len) tuples where
        full_ids is [1, prompt_len + gen_len] — same format as
        TransformersPolicy._generate_with_full_ids.
        """
        prompt_text = self._build_prompt(system_prompt, user_prompt)
        outputs = self.llm.generate(
            [prompt_text],
            sampling_params=self._sampling_params(K),
            lora_request=self._lora_req,
        )
        req = outputs[0]
        prompt_ids = list(req.prompt_token_ids)
        input_len  = len(prompt_ids)

        result = []
        for cand in req.outputs:   # K CompletionOutput objects
            gen_ids = list(cand.token_ids)
            full_ids = torch.tensor([prompt_ids + gen_ids], dtype=torch.long)
            result.append((cand.text, full_ids, input_len))
        return result

    def generate_one(
        self,
        system_prompt: str,
        user_prompt: str,
        use_lora: bool = False,
    ) -> str:
        """Generate one response (opponent turns — base model, no LoRA)."""
        prompt_text = self._build_prompt(system_prompt, user_prompt)
        outputs = self.llm.generate(
            [prompt_text],
            sampling_params=self._sampling_params(1),
            lora_request=(self._lora_req if use_lora else None),
        )
        return outputs[0].outputs[0].text


# ----------------------------------------------------------------------
# vLLM episode collector
# ----------------------------------------------------------------------

def collect_episode_vllm(
    game_id: str,
    num_players: int,
    vllm_engine: VLLMRolloutEngine,
    probe_policy,                    # TransformersPolicy on train GPU (probe only)
    probe_layer: int,
    reward_fn,
    grpo_k: int = 4,
    probe=None,
    system_prompt: str = "You are a strategic game player. Respond concisely.",
    max_turns: int = 50,
    verbose: bool = True,
) -> Episode:
    """Collect one episode using vLLM for all generation.

    Trainee K candidates:  ONE vLLM call (SamplingParams(n=K)) — fast.
    Opponent:              ONE vLLM call, no LoRA (base model).
    probe_text():          still runs on HF model (GPU 1, no generation).
    Returns the same Episode type as collect_episode().

    Note: probe_z / probe_z_all (trainee's OWN hidden state) are not
    available when generating with vLLM (no hook access).  These fields
    are left None; the adapter_a/inject breakdown still works because it
    runs a separate HF forward pass at log time.
    """
    import textarena as ta
    env = ta.make(game_id)
    env.reset(num_players=num_players)
    episode = Episode()
    turn_count = 0

    while turn_count < max_turns:
        player_id, obs_str = env.get_observation()

        if player_id == TRAINEE_ID:
            if verbose:
                print(f"    turn {turn_count+1} trainee "
                      f"(x{grpo_k}, vLLM batch)...", flush=True)
            t0 = time.time()

            # K candidates in ONE vLLM call — the core speedup
            candidates = vllm_engine.generate_candidates(
                system_prompt, obs_str, K=grpo_k
            )

            records: List[TurnRecord] = []
            rewards: List[float] = []

            for action, full_ids, input_len in candidates:
                opp_scores  = probe_policy.probe_text(action)
                ld_opp      = opp_scores.get(str(probe_layer), {})
                opp_z       = ld_opp.get("z") or None
                reward      = reward_fn(opp_scores) if opp_scores else 0.0

                records.append(TurnRecord(
                    obs=obs_str,
                    action=action,
                    full_ids=full_ids,
                    input_len=input_len,
                    log_prob=None,
                    probe_z=None,           # not available from vLLM
                    probe_z_opponent=opp_z,
                    probe_z_all=None,       # not available from vLLM
                ))
                rewards.append(reward)

            elapsed = time.time() - t0
            best_r  = max(rewards)
            mean_r  = sum(rewards) / len(rewards)

            if verbose:
                words = len(records[0].action.split())
                trait_str = ""
                if probe is not None:
                    best_k = rewards.index(best_r)
                    best_opp_z = records[best_k].probe_z_opponent
                    if best_opp_z:
                        try:
                            top = probe.rank_traits(best_opp_z, probe_layer)
                            if top:
                                trait_str = f"  top={top[0][0]}:{top[0][1]:+.2f}"
                        except Exception:
                            pass
                print(
                    f"    turn {turn_count+1} done — {words} words  "
                    f"{elapsed:.1f}s  "
                    f"r=[{mean_r:+.3f} mean, {best_r:+.3f} best]{trait_str}",
                    flush=True,
                )

            episode.turn_groups.append(
                TurnGroup(obs=obs_str, records=records, rewards=rewards)
            )
            done, _ = env.step(records[0].action)

        else:
            if verbose:
                print(f"    turn {turn_count+1} opponent (vLLM, base)...",
                      flush=True)
            t0 = time.time()
            action = vllm_engine.generate_one(
                system_prompt, obs_str, use_lora=False
            )
            elapsed = time.time() - t0
            if verbose:
                print(f"    turn {turn_count+1} done  {elapsed:.1f}s",
                      flush=True)
            done, _ = env.step(action)

        turn_count += 1
        if done:
            break

    game_rewards, _ = env.close()
    episode.game_rewards = {int(k): float(v) for k, v in game_rewards.items()}
    return episode
