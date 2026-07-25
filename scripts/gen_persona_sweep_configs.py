"""Generate configs for the behavioural-persona impact sweep.

21 conditions (1 plain baseline + 20 behavioural archetypes), all sharing the same
100 random targets (via seed_base) so that persona effects can be directly compared
across identical scenarios.

Game: gbs_exact_replication, 2 players, hide_group_size=True, directional feedback.
Model: Qwen3-14B (non-thinking).
Episodes: 100.

Usage
-----
python scripts/gen_persona_sweep_configs.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

OUT             = Path("config/persona_sweep")
PERSONAS_PATH   = Path("config/behavioral_personas.yaml")
MODEL_ID        = "Qwen/Qwen3-14B"
MODEL_TAG       = "14b"
NUM_PLAYERS     = 2
EPISODES        = 100
WANDB_PROJECT   = "ma-steering-persona-impact"
SEED_BASE       = "persona_sweep_shared"   # same targets across all conditions

OUT.mkdir(parents=True, exist_ok=True)


def load_personas() -> dict[str, str]:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["personas"]


def model_cfg() -> dict:
    return {
        "backend": "transformers",
        "model_id": MODEL_ID,
        "enable_thinking": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    }


def write_config(persona_key: str | None, persona_text: str | None) -> None:
    """Write one config.  persona_key=None → plain baseline."""
    condition = persona_key if persona_key is not None else "plain"
    run_id    = f"persona_impact_{condition}_2p_{MODEL_TAG}"

    env_kwargs: dict = {
        "num_rounds":      30,
        "low":             0,
        "feedback":        "directional",
        "hide_group_size": True,
        "persona_mode":    "persona" if persona_text is not None else "plain",
        "seed_base":       SEED_BASE,   # shared across conditions — same target per episode
    }
    if persona_text is not None:
        # Repeat the same persona for all N players so every agent plays the same role.
        env_kwargs["personas"] = [persona_text.strip()] * NUM_PLAYERS

    cfg = {
        "run_id": run_id,
        "game": {
            "family":    "symbolic",
            "id":        "gbs_exact_replication",
            "env_kwargs": env_kwargs,
        },
        "episodes": EPISODES,
        "model":    model_cfg(),
        "agents": {
            "count":             NUM_PLAYERS,
            "concurrency":       "sequential",
            "max_parse_retries": 5,
        },
        "steering": {"default": "noop", "per_agent": {}},
        "logging":  {"dir": "logs/persona_sweep/"},
        "wandb": {
            "enabled": True,
            "project": WANDB_PROJECT,
            "name":    run_id,
            "tags":    ["persona_sweep", condition, f"{NUM_PLAYERS}p", MODEL_TAG],
        },
    }

    path = OUT / f"{run_id}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  wrote {path}")


def main() -> None:
    personas = load_personas()
    print(f"Loaded {len(personas)} behavioural personas from {PERSONAS_PATH}")

    # Baseline — no persona injection
    write_config(None, None)

    # One config per behavioural persona
    for key, text in personas.items():
        write_config(key, text)

    total = 1 + len(personas)
    print(f"\nGenerated {total} configs -> {OUT}/")
    print(f"Shared seed_base: '{SEED_BASE}' — all conditions face the same 100 targets.")


if __name__ == "__main__":
    main()
