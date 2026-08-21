"""
Train dual persona LoRA adapters via GRPO self-play.

Two adapters are trained on the same trainee agent; both share the same reward
(SVD cosine-similarity to the target trait direction):

  adapter_a — persona LoRA: lora_B frozen to SVD trait directions, lora_A trainable.
              Can only activate pre-known persona axes — interpretable, constrained.
  adapter_b — standard LoRA on q/v/o_proj: free weights, finds its own path.
              Unconstrained baseline for comparison.

Both adapters are active during trainee turns; the opponent plays with both
disabled (base model only) via peft's disable_adapter() context.

Usage:
    python scripts/train_lora.py --config configs/training/persona_lora.yaml
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from collections import deque
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from testbed.policy.transformers_policy import TransformersPolicy
from testbed.probing.svd_probe import SVDPersonaProbe
from testbed.training.reward import PersonaReward
from testbed.training.rollout import collect_episode, recompute_logprobs, TRAINEE_ID, OPPONENT_ID
from testbed.training.grpo import grpo_step, episode_stats, wandb_log_step


def _init_wandb(cfg: dict):
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping. pip install wandb to enable.")
        return None
    return wandb.init(
        project=wcfg.get("project", "ma-steering-lora"),
        name=wcfg.get("name", None),
        tags=wcfg.get("tags", []),
        config=cfg,
        dir="wandb_logs",
    )


# ---------------------------------------------------------------------------
# Memory-efficient frozen opponent
# ---------------------------------------------------------------------------
class _FrozenOpponent:
    """Plays opponent turns using base weights (all adapters disabled).

    Reuses the trainee's PeftModel under peft's disable_adapter() context
    manager — no second model copy needed.
    """

    def __init__(self, peft_model, trainee_policy: TransformersPolicy):
        self._peft_model = peft_model
        self._policy = trainee_policy

    def act(self, system_prompt, user_prompt, agent_id, steering,
            return_logprob: bool = False):
        with self._peft_model.disable_adapter():
            return self._policy.act(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                agent_id=agent_id,
                steering=steering,
                return_logprob=return_logprob,
            )

    @property
    def _last_probe(self):
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _adapter_params(model, adapter_name: str):
    """Return the list of parameters belonging to a named LoRA adapter."""
    return [p for n, p in model.named_parameters()
            if adapter_name in n and p.requires_grad]


def _init_svd_lora_a(model, basis_path: str, adapter_name: str):
    """Initialize lora_B (up-projection) for o_proj from SVD basis Vk and freeze it.

    o_proj maps [4096 → 2560] in Qwen3-4B.  Vk lives in the 2560-dim OUTPUT
    space, so it aligns with lora_B (shape [d_out, r] = [2560, r]), not lora_A.

    After this call:
      lora_B columns = SVD trait directions  →  frozen  (up-projection)
      lora_A                                 →  trainable (down-projection)

    The adapter can only push activations along known persona axes; lora_A
    learns which input patterns activate which direction.
    """
    basis = torch.load(os.path.expandvars(basis_path), map_location="cpu",
                       weights_only=False)
    Vk = basis["Vk"]  # {layer_int: Tensor[k, d_out]}

    initialized = 0
    for name, param in model.named_parameters():
        if "lora_B" not in name or adapter_name not in name or "o_proj" not in name:
            continue
        parts = name.split(".")
        try:
            layer_idx = next(int(p) for p in parts if p.isdigit())
        except StopIteration:
            continue
        if layer_idx not in Vk:
            continue
        vk = Vk[layer_idx].float()  # [k, d_out]
        d_out, r = param.shape      # [d_out, r]
        k = vk.shape[0]
        with torch.no_grad():
            if r <= k:
                # B[:, i] = i-th SVD direction  →  B = Vk[:r].T
                param.data.copy_(vk[:r, :d_out].T)
            else:
                param.data[:d_out, :k].copy_(vk[:, :d_out].T)
                param.data[:, k:].zero_()
        param.requires_grad_(False)
        initialized += 1

    print(f"  SVD-init + froze {initialized} lora_B tensors for {adapter_name!r}"
          f" (o_proj only; lora_A remains trainable)")


def _set_adapters(model, adapter_names: list):
    """Activate one or more LoRA adapters, compatible across PEFT versions."""
    if len(adapter_names) == 1:
        model.set_adapter(adapter_names[0])
        return
    # Try list form first (PEFT >= 0.13 on PeftModel level may vary).
    try:
        model.set_adapter(adapter_names)
        return
    except TypeError:
        pass
    # Fallback: set via the underlying LoraModel directly.
    try:
        model.base_model.set_adapter(adapter_names)
        return
    except Exception:
        pass
    # Last resort: enable all adapter layers and set each module manually.
    model.enable_adapter_layers()
    for module in model.modules():
        if hasattr(module, "set_adapter"):
            try:
                module.set_adapter(adapter_names)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-persona-lora", action="store_true")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--traits", default=None,
                        help="Override target_traits: 'slug:weight,slug:weight' "
                             "(e.g. 'agreeableness:1.5,empathy:1.0')")
    parser.add_argument("--run-name", default=None,
                        help="Override wandb run name and output save_dir suffix")
    parser.add_argument("--local-save-dir", default=None,
                        help="Extra save path (e.g. scratch dir) for checkpoints and final adapters")
    parser.add_argument("--game", default=None,
                        help="Override training.game (e.g. SimpleNegotiation-v0)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override training.episodes (useful for quick smoke tests)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.no_persona_lora:
        cfg["use_persona_lora"] = False
    if args.rank is not None:
        cfg.setdefault("lora_a", {})["rank"] = args.rank
        cfg.setdefault("lora_b", {})["rank"] = args.rank
    if args.traits is not None:
        parsed = {}
        for part in args.traits.split(","):
            slug, weight = part.strip().split(":")
            parsed[slug.strip()] = float(weight.strip())
        cfg.setdefault("reward", {})["target_traits"] = parsed
        print(f"[CLI] target_traits override: {parsed}", flush=True)
    if args.run_name is not None:
        cfg.setdefault("wandb", {})["name"] = args.run_name
        base = cfg["output"]["save_dir"].rstrip("/")
        cfg["output"]["save_dir"] = f"{base}/{args.run_name}"
        print(f"[CLI] run_name={args.run_name}  save_dir={cfg['output']['save_dir']}",
              flush=True)

    if args.game is not None:
        cfg.setdefault("training", {})["game"] = args.game
        print(f"[CLI] game override: {args.game}", flush=True)
    if args.episodes is not None:
        cfg.setdefault("training", {})["episodes"] = args.episodes
        print(f"[CLI] episodes override: {args.episodes}", flush=True)

    local_save_dir = Path(args.local_save_dir) if args.local_save_dir else None
    if local_save_dir:
        run_suffix = args.run_name or cfg.get("wandb", {}).get("name", "run")
        local_save_dir = local_save_dir / run_suffix
        local_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CLI] local_save_dir={local_save_dir}", flush=True)

    use_persona_lora = cfg.get("use_persona_lora", True)

    save_dir = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load base model ──────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model, TaskType
    import peft as _peft
    from packaging.version import Version
    if Version(_peft.__version__) < Version("0.6.0"):
        raise RuntimeError(
            f"PEFT >= 0.6.0 required (installed: {_peft.__version__}).\n"
            f"  Fix: pip install 'peft>=0.6'"
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg["model"]["base"]
    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))

    print(f"Loading tokenizer for {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"Loading model weights ({cfg['model'].get('dtype', 'bfloat16')})...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                       device_map="auto")
    print("Model loaded.", flush=True)

    # ── Probe config (needed early for SVD init) ─────────────────────────────
    probe_cfg = cfg["probe"]

    # ── Attach LoRA adapters ─────────────────────────────────────────────────
    def _make_lora_config(lora_cfg):
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["targets"],
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
        )

    lora_cfg_b = _make_lora_config(cfg["lora_b"])

    print("Attaching LoRA adapters...", flush=True)
    if use_persona_lora:
        lora_cfg_a = _make_lora_config(cfg["lora_a"])
        model = get_peft_model(base_model, lora_cfg_a, adapter_name="adapter_a")
        print("  SVD-init adapter_a lora_B from basis...", flush=True)
        _init_svd_lora_a(model, probe_cfg["basis_path"], "adapter_a")
        model.add_adapter("adapter_b", lora_cfg_b)
        _set_adapters(model, ["adapter_a", "adapter_b"])
    else:
        print("  use_persona_lora=False — training adapter_b only", flush=True)
        model = get_peft_model(base_model, lora_cfg_b, adapter_name="adapter_b")
        _set_adapters(model, ["adapter_b"])

    # Gradient checkpointing: recompute activations during backward instead of
    # caching them.  Halves activation memory at ~20% extra compute cost.
    print("Enabling gradient checkpointing...", flush=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.train()
    print("Trainable parameters:", flush=True)
    model.print_trainable_parameters()

    # ── Build probe ──────────────────────────────────────────────────────────
    # PeftModel wraps Qwen3ForCausalLM as .model, so transformer layers sit at
    # .model.model.layers — one extra ".model" vs the bare Qwen3ForCausalLM case.
    # Probe: register hooks on ALL layers in the basis (residual stream).
    # reward_layer_start filters which layers contribute to the reward;
    # display_layer is used for rank_traits / wandb trait labels.
    probe = SVDPersonaProbe(
        basis_path=probe_cfg["basis_path"],
        layers=None,                        # all layers present in the basis
        hook=probe_cfg.get("hook", "residual"),
        top_k=cfg.get("top_k", 7),
        layer_path_template="model.model.layers.{}",
    )
    probe_layer = probe_cfg.get("display_layer") or max(probe.layers)

    # ── Build policies ───────────────────────────────────────────────────────
    max_new_tokens = cfg.get("training", {}).get("max_new_tokens", 1024)
    trainee_policy = TransformersPolicy(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        steering=None,
        probe=probe,
        max_new_tokens=max_new_tokens,
    )
    opponent_policy = _FrozenOpponent(model, trainee_policy)

    # ── Build reward ─────────────────────────────────────────────────────────
    # Both adapters share the same objective: maximise cosine-sim to target
    # traits.  adapter_a achieves this while constrained to the SVD persona
    # space; adapter_b is a free LoRA that finds its own path.
    reward_cfg = cfg["reward"]
    reward_alpha = float(reward_cfg.get("alpha", 1.0))

    persona_reward = PersonaReward(
        basis_path=probe_cfg["basis_path"],
        target_traits=reward_cfg["target_traits"],
        layer_start=probe_cfg.get("reward_layer_start", 10),
        sign=+1.0,
    )

    # ── Optimizers (one per adapter) ─────────────────────────────────────────
    opt_cfg = cfg.get("optimizer", {})
    lr = opt_cfg.get("lr", 1e-4)
    max_grad_norm = opt_cfg.get("max_grad_norm", 1.0)

    params_b = _adapter_params(model, "adapter_b")
    optimizer_b = torch.optim.AdamW(params_b, lr=lr)
    if use_persona_lora:
        params_a = _adapter_params(model, "adapter_a")
        optimizer_a = torch.optim.AdamW(params_a, lr=lr)
    else:
        optimizer_a = None

    # ── Training config ───────────────────────────────────────────────────────
    train_cfg = cfg["training"]
    game_id = train_cfg["game"]
    num_players = train_cfg.get("num_players", 2)
    grpo_k = train_cfg.get("grpo_k", 4)   # rollouts per GRPO update step
    save_interval = cfg["output"].get("save_interval", 50)

    wandb_run = _init_wandb(cfg)
    episode_log = []
    step = 0

    # ── Startup banner ────────────────────────────────────────────────────────
    target_traits = reward_cfg["target_traits"]
    target_trait_slugs = list(target_traits.keys())
    traits_str = ", ".join(f"{s}×{w:+g}" for s, w in target_traits.items())
    adapters_str = (
        f"adapter_a (SVD r={cfg['lora_a']['rank']}) + adapter_b (free r={cfg['lora_b']['rank']})"
        if use_persona_lora else f"adapter_b only (free r={cfg['lora_b']['rank']})"
    )
    total_episodes = train_cfg["episodes"]
    run_name = cfg.get("wandb", {}).get("name", "unnamed")
    print("", flush=True)
    print("═" * 60, flush=True)
    print(f"  Persona LoRA Training", flush=True)
    print(f"  run      : {run_name}", flush=True)
    print(f"  game     : {game_id}", flush=True)
    print(f"  model    : {model_id}", flush=True)
    print(f"  traits   : {traits_str}", flush=True)
    print(f"  adapters : {adapters_str}", flush=True)
    print(f"  episodes : {total_episodes}  grpo_k={grpo_k}  save_every={save_interval}", flush=True)
    print(f"  save_dir : {save_dir}", flush=True)
    if local_save_dir:
        print(f"  local    : {local_save_dir}", flush=True)
    print("═" * 60, flush=True)
    print("", flush=True)

    rolling_rewards: deque = deque(maxlen=10)
    train_start = time.time()

    def _reward_fn(opp_z):
        if opp_z is None:
            return 0.0
        return reward_alpha * persona_reward(opp_z)

    def _fmt_eta(elapsed: float, done: int, total: int) -> str:
        if done == 0:
            return "?"
        remaining = elapsed / done * (total - done)
        h, m = divmod(int(remaining), 3600)
        m //= 60
        return f"{h}h{m:02d}m" if h else f"{m}m"

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(1, total_episodes + 1):
        step_t0 = time.time()

        # Restore active adapters before each episode.
        if use_persona_lora:
            _set_adapters(model, ["adapter_a", "adapter_b"])
        else:
            _set_adapters(model, ["adapter_b"])

        print(f"[step {ep:4d}/{total_episodes}] collecting episode (k={grpo_k} candidates/turn)...",
              flush=True)

        episode = collect_episode(
            game_id=game_id,
            num_players=num_players,
            trainee_policy=trainee_policy,
            opponent_policy=opponent_policy,
            probe_layer=probe_layer,
            reward_fn=_reward_fn,
            grpo_k=grpo_k,
            probe=probe,
            max_turns=train_cfg.get("max_turns", 50),
        )

        n_turns = len(episode.turn_groups)
        all_rewards = [r for tg in episode.turn_groups for r in tg.rewards]
        mean_r = sum(all_rewards) / max(len(all_rewards), 1)
        rolling_rewards.append(mean_r)
        roll10 = sum(rolling_rewards) / len(rolling_rewards)

        # Game outcome for trainee
        trainee_game_r = episode.game_rewards.get(TRAINEE_ID, float("nan"))
        opponent_game_r = episode.game_rewards.get(OPPONENT_ID, float("nan"))
        game_outcome = {"trainee": trainee_game_r, "opponent": opponent_game_r}
        game_str = f"  game={trainee_game_r:+.2f}" if trainee_game_r == trainee_game_r else ""

        # Top opponent trait from last turn's best candidate
        trait_str = ""
        if episode.turn_groups:
            last_tg = episode.turn_groups[-1]
            best_k = max(range(len(last_tg.rewards)), key=lambda k: last_tg.rewards[k])
            opp_z = last_tg.records[best_k].probe_z_opponent
            if opp_z:
                top = probe.rank_traits(opp_z, probe_layer)
                if top:
                    trait_str = f"  top={top[0][0]}:{top[0][1]:+.2f}"
        print(f"  {n_turns} turns  r={mean_r:+.4f} (roll10={roll10:+.4f}){game_str}{trait_str}",
              flush=True)

        # Recompute log_probs WITH gradients right before backward.
        device = next(model.parameters()).device
        print(f"  recomputing log_probs ({n_turns * grpo_k} fwd passes)...", flush=True)
        recompute_logprobs(episode, model, str(device))

        # GRPO update — one backward per TurnGroup, advantages computed per turn.
        optimizers = [optimizer_a, optimizer_b] if use_persona_lora else [optimizer_b]
        combined_loss = grpo_step(episode, optimizers, max_grad_norm=max_grad_norm)

        stats = episode_stats(episode)
        step += 1
        step_time = time.time() - step_t0
        elapsed = time.time() - train_start
        eta = _fmt_eta(elapsed, step, total_episodes)

        entry = {
            "step": step,
            "loss": combined_loss,
            "reward": stats,
            "num_turns": n_turns,
            "game_reward_trainee": trainee_game_r,
            "step_time_s": step_time,
        }
        episode_log.append(entry)

        wandb_log_step(
            wandb_run=wandb_run,
            step=step,
            episode=episode,
            loss=combined_loss,
            model=model,
            probe=probe,
            probe_layer=probe_layer,
            use_persona_lora=use_persona_lora,
            log_transcript_every=cfg["output"].get("log_transcript_every", 50),
            rolling_mean=roll10,
            game_outcome=game_outcome,
            step_time=step_time,
            target_trait_slugs=target_trait_slugs,
        )

        print(
            f"  ✓ step {step:4d}  loss={combined_loss:.4f}  "
            f"r={stats['mean']:+.4f}±{stats['std']:.3f}  "
            f"roll10={roll10:+.4f}  {step_time:.0f}s  ETA {eta}",
            flush=True,
        )

        if ep % save_interval == 0:
            ckpt = save_dir / f"checkpoint_step{step:05d}"
            if use_persona_lora:
                model.save_pretrained(str(ckpt / "adapter_a"),
                                      selected_adapters=["adapter_a"])
            model.save_pretrained(str(ckpt / "adapter_b"),
                                  selected_adapters=["adapter_b"])
            tokenizer.save_pretrained(str(ckpt))
            print(f"  Saved checkpoint → {ckpt}", flush=True)
            if local_save_dir:
                local_ckpt = local_save_dir / f"checkpoint_step{step:05d}"
                shutil.copytree(str(ckpt), str(local_ckpt), dirs_exist_ok=True)
                print(f"  Saved checkpoint → {local_ckpt} (local)", flush=True)

    # ── Final save ────────────────────────────────────────────────────────────
    final_dir = save_dir / "final"
    if use_persona_lora:
        model.save_pretrained(str(final_dir / "adapter_a"),
                              selected_adapters=["adapter_a"])
    model.save_pretrained(str(final_dir / "adapter_b"),
                          selected_adapters=["adapter_b"])
    tokenizer.save_pretrained(str(final_dir))
    with open(save_dir / "training_log.jsonl", "w") as f:
        for entry in episode_log:
            f.write(json.dumps(entry) + "\n")

    if local_save_dir:
        local_final = local_save_dir / "final"
        shutil.copytree(str(final_dir), str(local_final), dirs_exist_ok=True)
        shutil.copy(str(save_dir / "training_log.jsonl"),
                    str(local_save_dir / "training_log.jsonl"))
        print(f"  Saved final → {local_final} (local)", flush=True)

    total_time = time.time() - train_start
    h, rem = divmod(int(total_time), 3600)
    m = rem // 60
    print(f"\nTraining complete in {h}h{m:02d}m. Outputs in {save_dir}")
    if local_save_dir:
        print(f"Local copy: {local_save_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
