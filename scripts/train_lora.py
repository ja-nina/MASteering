"""
Train a persona or task LoRA via REINFORCE self-play.

Usage:
    python scripts/train_lora.py --config configs/training/persona_lora.yaml
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from testbed.policy.transformers_policy import TransformersPolicy
from testbed.probing.svd_probe import SVDPersonaProbe
from testbed.training.reward import PersonaReward, TaskReward, CombinedReward
from testbed.training.rollout import collect_episode, TRAINEE_ID
from testbed.training.reinforce import reinforce_step


# ---------------------------------------------------------------------------
# Memory-efficient frozen opponent: reuses the trainee's LoRA model with all
# adapters disabled so we only keep one copy of the weights in GPU memory.
# ---------------------------------------------------------------------------
class _FrozenOpponent:
    """Plays the opponent role using base (non-LoRA) weights.

    Wraps the trainee policy and temporarily disables all adapters via
    peft's ``disable_adapter()`` context manager, avoiding the cost of
    loading a second copy of the base model.
    """

    def __init__(self, lora_model, trainee_policy: TransformersPolicy):
        self._model = lora_model        # PeftModel — for disable_adapter()
        self._policy = trainee_policy

    def act(self, system_prompt, user_prompt, agent_id, steering,
            return_logprob: bool = False):
        with self._model.disable_adapter():
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
# Reward construction
# ---------------------------------------------------------------------------
def build_reward(cfg: dict):
    mode = cfg["reward"]["mode"]
    basis = cfg["probe"]["basis_path"]
    layer = cfg["probe"]["layer"]
    if mode == "persona":
        return PersonaReward(basis, layer, cfg["reward"]["target_traits"])
    elif mode == "task":
        return TaskReward()
    else:  # combined
        p = PersonaReward(basis, layer, cfg["reward"]["target_traits"])
        t = TaskReward()
        return CombinedReward(p, t,
            cfg["reward"].get("persona_weight", 1.0),
            cfg["reward"].get("task_weight", 1.0))


def get_reward_value(reward_fn, record, game_rewards):
    if isinstance(reward_fn, PersonaReward):
        return reward_fn(record.probe_z) if record.probe_z else 0.0
    elif isinstance(reward_fn, TaskReward):
        return reward_fn(game_rewards, TRAINEE_ID)
    else:  # CombinedReward
        return reward_fn(record.probe_z, game_rewards, TRAINEE_ID)


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

    # ── Load model + LoRA ────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = cfg["model"]["base"]
    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                       device_map="auto")

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["targets"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    # Set train mode so LoRA params receive gradients; base weights stay frozen.
    model.train()

    # ── Build probe ──────────────────────────────────────────────────────────
    probe_cfg = cfg["probe"]
    probe = SVDPersonaProbe(
        basis_path=probe_cfg["basis_path"],
        layers=[probe_cfg["layer"]],
        hook=probe_cfg.get("hook", "attn"),
        top_k=cfg.get("top_k", 7),
    )

    # ── Build policies ───────────────────────────────────────────────────────
    # The trainee policy receives the injected LoRA model; caller (here) owns
    # train/eval mode so we do not call model.eval() inside __init__.
    trainee_policy = TransformersPolicy(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        steering=None,
        probe=probe,
    )

    # The opponent reuses the same model weights with adapters disabled —
    # no second model load needed.
    opponent_policy = _FrozenOpponent(model, trainee_policy)

    # ── Optimizer ────────────────────────────────────────────────────────────
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg.get("lr", 1e-4),
    )

    reward_fn = build_reward(cfg)
    train_cfg = cfg["training"]
    probe_layer = probe_cfg["layer"]
    game_id = train_cfg["game"]
    num_players = train_cfg.get("num_players", 2)
    max_grad_norm = opt_cfg.get("max_grad_norm", 1.0)

    # ── Training loop ────────────────────────────────────────────────────────
    baseline = 0.0
    log_interval = cfg["output"].get("log_interval", 10)
    save_interval = cfg["output"].get("save_interval", 50)
    episode_log = []

    for ep in range(1, train_cfg["episodes"] + 1):
        episode = collect_episode(
            game_id=game_id,
            num_players=num_players,
            trainee_policy=trainee_policy,
            opponent_policy=opponent_policy,
            reward_fn=reward_fn,
            probe_layer=probe_layer,
            max_turns=train_cfg.get("max_turns", 50),
        )

        rewards = [get_reward_value(reward_fn, r, episode.game_rewards)
                   for r in episode.records]

        loss, baseline = reinforce_step(
            episode.records, rewards, optimizer, baseline,
            gamma=train_cfg.get("gamma", 0.99),
            max_grad_norm=max_grad_norm,
        )

        mean_r = sum(rewards) / max(len(rewards), 1)
        mean_z_score = None
        if episode.records:
            zs = [r.probe_z for r in episode.records if r.probe_z]
            if zs and isinstance(reward_fn, PersonaReward):
                mean_z_score = sum(reward_fn(z) for z in zs) / len(zs)

        entry = {"episode": ep, "loss": loss, "mean_reward": mean_r,
                 "mean_persona_score": mean_z_score, "baseline": baseline,
                 "n_turns": len(episode.records)}
        episode_log.append(entry)

        if ep % log_interval == 0:
            z_str = f"{mean_z_score:.4f}" if mean_z_score is not None else "n/a"
            print(f"[ep {ep:4d}] loss={loss:.4f}  mean_r={mean_r:.4f}  "
                  f"persona={z_str}  turns={len(episode.records)}")

        if ep % save_interval == 0:
            ckpt = save_dir / f"checkpoint_ep{ep:05d}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            print(f"  Saved checkpoint -> {ckpt}")

    # Final save
    model.save_pretrained(str(save_dir / "final"))
    tokenizer.save_pretrained(str(save_dir / "final"))
    with open(save_dir / "training_log.jsonl", "w") as f:
        for entry in episode_log:
            f.write(json.dumps(entry) + "\n")
    print(f"\nTraining complete. Outputs in {save_dir}")


if __name__ == "__main__":
    main()
