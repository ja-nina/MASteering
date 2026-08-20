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
import argparse, json, os, sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from testbed.policy.transformers_policy import TransformersPolicy
from testbed.probing.svd_probe import SVDPersonaProbe
from testbed.training.reward import PersonaReward
from testbed.training.rollout import collect_episode, recompute_logprobs, TRAINEE_ID
from testbed.training.grpo import grpo_step, group_stats, wandb_log_step


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
    parser.add_argument("--no-persona-lora", action="store_true",
                        help="Skip adapter_a entirely; train only the minimiser (adapter_b)")
    parser.add_argument("--rank", type=int, default=None,
                        help="Override lora rank for both adapters")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.no_persona_lora:
        cfg["use_persona_lora"] = False
    if args.rank is not None:
        cfg.setdefault("lora_a", {})["rank"] = args.rank
        cfg.setdefault("lora_b", {})["rank"] = args.rank

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
    probe = SVDPersonaProbe(
        basis_path=probe_cfg["basis_path"],
        layers=[probe_cfg["layer"]],
        hook=probe_cfg.get("hook", "attn"),
        top_k=cfg.get("top_k", 7),
        layer_path_template="model.model.layers.{}",
    )

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
    basis_path = probe_cfg["basis_path"]
    probe_layer = probe_cfg["layer"]

    persona_reward = PersonaReward(
        basis_path=basis_path,
        layer=probe_layer,
        target_traits=reward_cfg["target_traits"],
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
    log_interval = cfg["output"].get("log_interval", 10)
    save_interval = cfg["output"].get("save_interval", 50)

    wandb_run = _init_wandb(cfg)
    episode_log = []
    step = 0  # counts GRPO update steps (each step = grpo_k episodes)

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(1, train_cfg["episodes"] + 1):
        # Restore active adapters before each episode.
        if use_persona_lora:
            _set_adapters(model, ["adapter_a", "adapter_b"])
        else:
            _set_adapters(model, ["adapter_b"])

        # Collect a group of K episodes for GRPO.
        group_episodes = []
        group_rewards_a = []
        group_rewards_b = []
        total_turns = 0

        print(f"[step {ep:4d}] collecting {grpo_k} episodes...", flush=True)
        for k in range(grpo_k):
            episode = collect_episode(
                game_id=game_id,
                num_players=num_players,
                trainee_policy=trainee_policy,
                opponent_policy=opponent_policy,
                probe_layer=probe_layer,
                max_turns=train_cfg.get("max_turns", 50),
            )
            # Same reward for both adapters — both push toward target traits.
            rewards = [persona_reward(r.probe_z) if r.probe_z else 0.0
                       for r in episode.records]
            ep_mean_r = sum(rewards) / max(len(rewards), 1)
            print(f"  ep {k+1}/{grpo_k}: {len(episode.records)} turns  "
                  f"r={ep_mean_r:+.4f}", flush=True)
            group_episodes.append(episode)
            group_rewards_a.append(rewards)
            group_rewards_b.append(rewards)
            total_turns += len(episode.records)

        # Recompute log_probs WITH gradients — happens once, right before backward.
        # Collection above was pure no_grad so no graphs were held during rollouts.
        device = next(model.parameters()).device
        print(f"  recomputing log_probs for {grpo_k} episodes...", flush=True)
        recompute_logprobs(group_episodes, model, str(device))

        # GRPO update — single call, episode-by-episode accumulation.
        group_records = [ep.records for ep in group_episodes]
        optimizers = [optimizer_a, optimizer_b] if use_persona_lora else [optimizer_b]
        combined_loss = grpo_step(group_records, group_rewards_a, optimizers,
                                  max_grad_norm=max_grad_norm)
        # Attribute loss equally for logging (same backward, same signal)
        loss_a = combined_loss if use_persona_lora else 0.0
        loss_b = combined_loss

        stats_a = group_stats(group_rewards_a)
        stats_b = group_stats(group_rewards_b)
        step += 1

        entry = {
            "step": step, "episodes": ep * grpo_k,
            "loss_a": loss_a, "loss_b": loss_b,
            "reward_a": stats_a, "reward_b": stats_b,
            "mean_turns": total_turns / grpo_k,
        }
        episode_log.append(entry)

        wandb_log_step(
            wandb_run=wandb_run,
            step=step,
            episodes=group_episodes,
            rewards_a=group_rewards_a,
            rewards_b=group_rewards_b,
            loss_a=loss_a,
            loss_b=loss_b,
            model=model,
            probe=probe,
            probe_layer=probe_layer,
            use_persona_lora=use_persona_lora,
            log_transcript_every=cfg["output"].get("log_transcript_every", 50),
        )

        if ep % log_interval == 0:
            print(f"[step {step:4d}] "
                  f"r_a={stats_a['mean']:+.4f}±{stats_a['std']:.3f}  loss_a={loss_a:.4f}  |  "
                  f"r_b={stats_b['mean']:+.4f}±{stats_b['std']:.3f}  loss_b={loss_b:.4f}  "
                  f"turns/ep={total_turns/grpo_k:.1f}")

        if ep % save_interval == 0:
            ckpt = save_dir / f"checkpoint_step{step:05d}"
            if use_persona_lora:
                model.save_pretrained(str(ckpt / "adapter_a"),
                                      selected_adapters=["adapter_a"])
            model.save_pretrained(str(ckpt / "adapter_b"),
                                  selected_adapters=["adapter_b"])
            tokenizer.save_pretrained(str(ckpt))
            print(f"  Saved checkpoint -> {ckpt}")

    # ── Final save ────────────────────────────────────────────────────────────
    if use_persona_lora:
        model.save_pretrained(str(save_dir / "final" / "adapter_a"),
                              selected_adapters=["adapter_a"])
    model.save_pretrained(str(save_dir / "final" / "adapter_b"),
                          selected_adapters=["adapter_b"])
    tokenizer.save_pretrained(str(save_dir / "final"))
    with open(save_dir / "training_log.jsonl", "w") as f:
        for entry in episode_log:
            f.write(json.dumps(entry) + "\n")
    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nTraining complete. Outputs in {save_dir}")


if __name__ == "__main__":
    main()
