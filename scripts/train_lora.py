"""
Train dual persona LoRA adapters via REINFORCE self-play.

Two adapters are trained simultaneously on the same trainee agent:
  adapter_a — maximiser: rewarded for high cosine sim to the maximize_traits direction.
  adapter_b — minimiser: rewarded for low  cosine sim to the minimize_traits direction
              (equivalently, high cosine sim to the negated direction; sign=-1 in reward).

Both adapters are active during trainee turns.  The opponent plays with both
disabled (base model weights only), using peft's disable_adapter() context so
only one copy of the weights lives in GPU memory.

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
from testbed.training.rollout import collect_episode, TRAINEE_ID
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg["model"]["base"]
    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                       device_map="auto")

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

    if use_persona_lora:
        lora_cfg_a = _make_lora_config(cfg["lora_a"])
        model = get_peft_model(base_model, lora_cfg_a, adapter_name="adapter_a")
        # SVD-init adapter_a: freeze lora_A (o_proj) to Vk; only lora_B trains.
        _init_svd_lora_a(model, probe_cfg["basis_path"], "adapter_a")
        model.add_adapter("adapter_b", lora_cfg_b)
        model.set_adapter(["adapter_a", "adapter_b"])
    else:
        print("  use_persona_lora=False — training minimiser (adapter_b) only")
        model = get_peft_model(base_model, lora_cfg_b, adapter_name="adapter_b")
        model.set_adapter(["adapter_b"])

    model.train()
    print("Trainable parameters:")
    model.print_trainable_parameters()

    # ── Build probe ──────────────────────────────────────────────────────────
    probe = SVDPersonaProbe(
        basis_path=probe_cfg["basis_path"],
        layers=[probe_cfg["layer"]],
        hook=probe_cfg.get("hook", "attn"),
        top_k=cfg.get("top_k", 7),
    )

    # ── Build policies ───────────────────────────────────────────────────────
    trainee_policy = TransformersPolicy(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        steering=None,
        probe=probe,
    )
    opponent_policy = _FrozenOpponent(model, trainee_policy)

    # ── Build rewards ────────────────────────────────────────────────────────
    reward_cfg = cfg["reward"]
    basis_path = probe_cfg["basis_path"]
    probe_layer = probe_cfg["layer"]

    reward_a = PersonaReward(
        basis_path=basis_path,
        layer=probe_layer,
        target_traits=reward_cfg["maximize_traits"],
        sign=+1.0,
    )
    reward_b = PersonaReward(
        basis_path=basis_path,
        layer=probe_layer,
        target_traits=reward_cfg["minimize_traits"],
        sign=-1.0,
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
            model.set_adapter(["adapter_a", "adapter_b"])
        else:
            model.set_adapter(["adapter_b"])

        # Collect a group of K episodes for GRPO.
        group_episodes = []
        group_rewards_a = []
        group_rewards_b = []
        total_turns = 0

        for _ in range(grpo_k):
            episode = collect_episode(
                game_id=game_id,
                num_players=num_players,
                trainee_policy=trainee_policy,
                opponent_policy=opponent_policy,
                probe_layer=probe_layer,
                max_turns=train_cfg.get("max_turns", 50),
            )
            ra = [reward_a(r.probe_z) if r.probe_z else 0.0 for r in episode.records]
            rb = [reward_b(r.probe_z) if r.probe_z else 0.0 for r in episode.records]
            group_episodes.append(episode)
            group_rewards_a.append(ra)
            group_rewards_b.append(rb)
            total_turns += len(episode.records)

        # GRPO update — adapter_a first (retain graph), then adapter_b.
        group_records = [ep.records for ep in group_episodes]
        if use_persona_lora:
            loss_a = grpo_step(group_records, group_rewards_a, optimizer_a,
                               max_grad_norm=max_grad_norm, retain_graph=True)
        else:
            loss_a = 0.0

        loss_b = grpo_step(group_records, group_rewards_b, optimizer_b,
                           max_grad_norm=max_grad_norm, retain_graph=False)

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
