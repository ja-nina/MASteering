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
from testbed.training.reinforce import reinforce_step


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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

    # ── Attach two LoRA adapters ─────────────────────────────────────────────
    def _make_lora_config(lora_cfg):
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["targets"],
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias="none",
        )

    lora_cfg_a = _make_lora_config(cfg["lora_a"])
    lora_cfg_b = _make_lora_config(cfg["lora_b"])

    # First adapter via get_peft_model, second via add_adapter.
    model = get_peft_model(base_model, lora_cfg_a, adapter_name="adapter_a")
    model.add_adapter("adapter_b", lora_cfg_b)
    # Activate both adapters for trainee turns.
    model.set_adapter(["adapter_a", "adapter_b"])
    model.train()

    print("Trainable parameters:")
    model.print_trainable_parameters()

    # ── Build probe ──────────────────────────────────────────────────────────
    probe_cfg = cfg["probe"]
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

    params_a = _adapter_params(model, "adapter_a")
    params_b = _adapter_params(model, "adapter_b")
    optimizer_a = torch.optim.AdamW(params_a, lr=lr)
    optimizer_b = torch.optim.AdamW(params_b, lr=lr)

    # ── Training config ───────────────────────────────────────────────────────
    train_cfg = cfg["training"]
    game_id = train_cfg["game"]
    num_players = train_cfg.get("num_players", 2)
    gamma = train_cfg.get("gamma", 0.99)
    log_interval = cfg["output"].get("log_interval", 10)
    save_interval = cfg["output"].get("save_interval", 50)

    baseline_a = 0.0
    baseline_b = 0.0
    episode_log = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(1, train_cfg["episodes"] + 1):
        # Ensure both adapters are active before each episode.
        model.set_adapter(["adapter_a", "adapter_b"])

        episode = collect_episode(
            game_id=game_id,
            num_players=num_players,
            trainee_policy=trainee_policy,
            opponent_policy=opponent_policy,
            probe_layer=probe_layer,
            max_turns=train_cfg.get("max_turns", 50),
        )

        # Compute per-turn rewards from probe z-vectors.
        rewards_a = [reward_a(r.probe_z) if r.probe_z else 0.0
                     for r in episode.records]
        rewards_b = [reward_b(r.probe_z) if r.probe_z else 0.0
                     for r in episode.records]

        # Two REINFORCE steps — use retain_graph=True for adapter_a so the
        # computation graph is still alive when adapter_b's step runs.
        loss_a, baseline_a = reinforce_step(
            episode.records, rewards_a, optimizer_a, baseline_a,
            gamma=gamma, max_grad_norm=max_grad_norm,
            retain_graph=True,   # keep graph for adapter_b's backward
        )
        loss_b, baseline_b = reinforce_step(
            episode.records, rewards_b, optimizer_b, baseline_b,
            gamma=gamma, max_grad_norm=max_grad_norm,
            retain_graph=False,
        )

        mean_ra = sum(rewards_a) / max(len(rewards_a), 1)
        mean_rb = sum(rewards_b) / max(len(rewards_b), 1)
        entry = {
            "episode": ep,
            "loss_a": loss_a, "loss_b": loss_b,
            "mean_reward_a": mean_ra, "mean_reward_b": mean_rb,
            "baseline_a": baseline_a, "baseline_b": baseline_b,
            "n_turns": len(episode.records),
        }
        episode_log.append(entry)

        if ep % log_interval == 0:
            print(f"[ep {ep:4d}] "
                  f"loss_a={loss_a:.4f}  r_a={mean_ra:+.4f}  "
                  f"loss_b={loss_b:.4f}  r_b={mean_rb:+.4f}  "
                  f"turns={len(episode.records)}")

        if ep % save_interval == 0:
            ckpt = save_dir / f"checkpoint_ep{ep:05d}"
            model.save_pretrained(str(ckpt / "adapter_a"), selected_adapters=["adapter_a"])
            model.save_pretrained(str(ckpt / "adapter_b"), selected_adapters=["adapter_b"])
            tokenizer.save_pretrained(str(ckpt))
            print(f"  Saved checkpoint -> {ckpt}")

    # ── Final save ────────────────────────────────────────────────────────────
    model.save_pretrained(str(save_dir / "final" / "adapter_a"),
                          selected_adapters=["adapter_a"])
    model.save_pretrained(str(save_dir / "final" / "adapter_b"),
                          selected_adapters=["adapter_b"])
    tokenizer.save_pretrained(str(save_dir / "final"))
    with open(save_dir / "training_log.jsonl", "w") as f:
        for entry in episode_log:
            f.write(json.dumps(entry) + "\n")
    print(f"\nTraining complete. Outputs in {save_dir}")


if __name__ == "__main__":
    main()
