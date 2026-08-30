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

import copy
import time
from typing import Dict, List, Optional, Tuple

import torch

from testbed.training.rollout import Episode, TurnGroup, TurnRecord, TRAINEE_ID, OPPONENT_ID
from testbed.training.generation_utils import FORMAT_REGEX, STRUCTURED_FORMAT_INSTRUCTION, _extract_action, _has_action_tag


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
        use_guided_decoding: bool = False,
    ):
        from vllm import LLM

        self.model_id   = model_id
        self.temperature          = temperature
        self.top_p               = top_p
        self.top_k               = top_k
        self.max_tokens          = max_tokens
        self.use_guided_decoding = use_guided_decoding

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
            enforce_eager=True,   # skip torch.compile; avoids flashinfer/py3.11 bug
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
            {"role": "system", "content": system_prompt + STRUCTURED_FORMAT_INSTRUCTION},
            {"role": "user",   "content": user_prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,   # disable Qwen3 native <think> mode
            )
        except TypeError:
            # older tokenizer version without enable_thinking
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def _sampling_params(self, n: int):
        from vllm import SamplingParams
        kwargs = dict(
            n=n,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        )
        if self.use_guided_decoding:
            # Try the new GuidedDecodingParams API first.
            try:
                from vllm.sampling_params import GuidedDecodingParams
                kwargs["guided_decoding"] = GuidedDecodingParams(regex=FORMAT_REGEX)
            except (ImportError, TypeError, AttributeError):
                # GuidedDecodingParams unavailable — try the older guided_regex kwarg.
                # Pass it directly at construction time so SamplingParams can reject it
                # cleanly without leaving a bad key in kwargs.
                try:
                    return SamplingParams(**kwargs, guided_regex=FORMAT_REGEX)
                except TypeError:
                    # Neither interface available in this vLLM build; fall through unguided.
                    print("[vLLM] WARNING: guided decoding unavailable in this build; "
                          "--bind has no effect", flush=True)
        return SamplingParams(**kwargs)

    def generate_candidates(
        self,
        system_prompt: str,
        user_prompt: str,
        K: int,
    ) -> List[Tuple[str, torch.Tensor, int]]:
        """Two-phase constrained generation of K trainee candidates.

        Phase 1: ONE vLLM call (n=K) with <strategy> forced as the prompt
                 suffix and stop=["</strategy>"] — K strategy blocks generated
                 in parallel, each stopping the moment </strategy> appears.
        Phase 2: K prompts (one per strategy) each ending with \\n<action>
                 batched in a single vLLM call (n=1 each) to generate the
                 action bodies.

        Returns list of K (text, full_ids_cpu, input_len) tuples where
        full_ids is [1, p1_prompt_len + strat_len + action_tag_len + action_len]
        and input_len = p1_prompt_len (prompt + forced <strategy> prefix), so
        log-prob computation covers the full strategy + injected <action> + action.
        """
        from vllm import SamplingParams

        base_prompt = self._build_prompt(system_prompt, user_prompt)
        # Force <strategy> as the start of the assistant turn
        p1_prompt = base_prompt + "<strategy>"

        # ── Phase 1: K strategy blocks, stop at </strategy> ──────────────────
        sp1_kwargs = dict(
            n=K,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            stop=["</strategy>"],
            include_stop_str_in_output=True,
        )
        out1 = self.llm.generate(
            [p1_prompt],
            sampling_params=SamplingParams(**sp1_kwargs),
            lora_request=self._lora_req,
        )
        req1 = out1[0]
        p1_prompt_ids = list(req1.prompt_token_ids)

        strategies: List[str] = []
        for cand in req1.outputs:
            s = cand.text
            if "</strategy>" not in s:
                s = s.rstrip() + "\n</strategy>"
            strategies.append(s)

        # ── Phase 2: inject \n<action>, batch K action bodies ─────────────────
        p2_prompts = [p1_prompt + s + "\n<action>" for s in strategies]
        sp2_kwargs = dict(
            n=1,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        )
        out2 = self.llm.generate(
            p2_prompts,
            sampling_params=SamplingParams(**sp2_kwargs),
            lora_request=self._lora_req,
        )

        # ── Reconstruct full token sequences ──────────────────────────────────
        action_tag_ids = self.tokenizer.encode("\n<action>", add_special_tokens=False)
        result: List[Tuple[str, torch.Tensor, int]] = []
        for i, (strat, o2_out) in enumerate(zip(strategies, out2)):
            action_body = o2_out.outputs[0].text
            full_text   = f"<strategy>{strat}\n<action>{action_body}"
            strat_ids   = list(req1.outputs[i].token_ids)
            action_ids  = list(o2_out.outputs[0].token_ids)
            full_ids    = torch.tensor(
                [p1_prompt_ids + strat_ids + action_tag_ids + action_ids],
                dtype=torch.long,
            )
            # input_len = p1_prompt_len so log probs cover strat+tag+action
            result.append((full_text, full_ids, len(p1_prompt_ids)))
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

    def generate_batch_prompts(
        self,
        prompt_pairs: List[Tuple[str, str]],
    ) -> List[str]:
        """Generate one response per (system_prompt, user_prompt) pair.

        Used for counterfactual opponent responses: given K trainee actions,
        produce K opponent replies in a single vLLM call (no LoRA).
        """
        texts = [self._build_prompt(sp, up) for sp, up in prompt_pairs]
        outputs = self.llm.generate(
            texts,
            sampling_params=self._sampling_params(1),
            lora_request=None,
        )
        return [out.outputs[0].text for out in outputs]


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
    trainee_system_prompt: Optional[str] = None,
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

    trainee_system_prompt: when set, used for trainee turns instead of
    system_prompt (which goes to the opponent).  Enables secret nudge
    instructions unknown to the opponent.
    """
    import textarena as ta
    env = ta.make(game_id)
    _trainee_sp = trainee_system_prompt if trainee_system_prompt is not None else system_prompt
    env.reset(num_players=num_players)
    episode = Episode()
    episode.system_prompt = system_prompt
    episode.trainee_system_prompt = trainee_system_prompt
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
                _trainee_sp, obs_str, K=grpo_k
            )

            # Counterfactual opponent probing: for each candidate, deep-copy the
            # env and step with that candidate's extracted action to get the
            # opponent's resulting observation, then batch-generate K opponent
            # responses and probe each one.  This measures the opponent's actual
            # hidden state when RESPONDING to the trainee — not just when reading.
            opp_prompt_pairs: List[Optional[Tuple[str, str]]] = []
            for action, _, _ in candidates:
                try:
                    env_cf = copy.deepcopy(env)
                    done_cf, _ = env_cf.step(_extract_action(action))
                    if done_cf:
                        opp_prompt_pairs.append(None)
                    else:
                        _, opp_obs_cf = env_cf.get_observation()
                        opp_prompt_pairs.append((system_prompt, opp_obs_cf))
                except Exception:
                    opp_prompt_pairs.append(None)

            valid_idxs = [i for i, p in enumerate(opp_prompt_pairs) if p is not None]
            if valid_idxs:
                batch_pairs = [opp_prompt_pairs[i] for i in valid_idxs]
                batch_opp_responses = vllm_engine.generate_batch_prompts(batch_pairs)
                opp_resp_map = dict(zip(valid_idxs, batch_opp_responses))
            else:
                opp_resp_map = {}

            records: List[TurnRecord] = []
            rewards: List[float] = []

            for k, (action, full_ids, input_len) in enumerate(candidates):
                opp_resp = opp_resp_map.get(k)
                # Probe the opponent's generated response (strategy + action) if
                # available; fall back to trainee's action if the game ended early.
                probe_input = opp_resp if opp_resp is not None else _extract_action(action)
                opp_scores  = probe_policy.probe_text(probe_input)
                ld_opp      = opp_scores.get(str(probe_layer), {})
                opp_z       = ld_opp.get("z") or None
                reward      = (reward_fn(opp_scores)
                               if (opp_scores and _has_action_tag(action)) else -1.0)

                # Opponent game-decision (cooperate/defect) — only meaningful in
                # the decision round.  Gate on the trainee also having decided;
                # communication-round responses mention these words incidentally.
                opp_decision = None
                trainee_act_lower = _extract_action(action).lower()
                if opp_resp is not None and (
                    "cooperat" in trainee_act_lower or "defect" in trainee_act_lower
                ):
                    opp_decision = _extract_action(opp_resp).lower()

                records.append(TurnRecord(
                    obs=obs_str,
                    action=action,
                    full_ids=full_ids,
                    input_len=input_len,
                    log_prob=None,
                    probe_z=None,           # not available from vLLM
                    probe_z_opponent=opp_z,
                    probe_z_all=None,       # not available from vLLM
                    opp_decision=opp_decision,
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
            episode.episode_log.append({"player": "trainee", "obs": obs_str,
                                        "action": records[0].action})
            done, _ = env.step(_extract_action(records[0].action))

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
            episode.episode_log.append({"player": "opponent", "obs": obs_str,
                                        "action": action})
            done, _ = env.step(_extract_action(action))

        turn_count += 1
        if done:
            break

    ta_rewards, _ = env.close()
    # IPD (and other length-capped games) return None when max_turns is hit
    # without a terminal done signal; game outcome is irrelevant to probe reward.
    if ta_rewards is not None:
        episode.game_rewards = {int(k): float(v) for k, v in ta_rewards.items()}
    else:
        episode.game_rewards = {i: 0.0 for i in range(num_players)}
    return episode
