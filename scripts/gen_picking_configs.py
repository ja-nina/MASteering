"""Generate configs for the Picking / Persona sweep (Riedl 2025, arXiv 2510.05174).

Three experimental conditions:
  plain   — binary search instructions only, no persona
  persona — persona prepended to system prompt
  tom     — persona + Theory-of-Mind instruction

Two models, three player counts.  All use:
  game_id         = picking  (same GBSAdapter, hide_group_size=True)
  feedback        = directional  (paper: agents only learn too HIGH / too LOW)
  low             = 0, high defaults to 50*N  (paper: each agent guesses from [0, 50])
  num_rounds      = 30  (paper imposes no hard cap; 30 is a practical upper bound)
  episodes        = 200  (matching paper's 200 replications per condition)

Usage
-----
python scripts/gen_picking_configs.py
"""
from pathlib import Path

import yaml

OUT = Path("config/picking_sweep")
OUT.mkdir(parents=True, exist_ok=True)

PERSONAS_PATH = Path("config/personas.yaml")

CONDITIONS = ["plain", "persona", "tom"]
PLAYERS    = [2, 3, 10]

WANDB_PROJECT = "ma-steering-picking"


def load_personas() -> list[str]:
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["personas"]


def model_cfg(model_id: str) -> dict:
    base = {"backend": "transformers", "model_id": model_id}
    if "Qwen3" in model_id:
        # Qwen3-14B non-thinking official recommendations (model card + tech report):
        # temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5
        # (presence_penalty is additive; repetition_penalty is multiplicative — different)
        return {**base, "enable_thinking": False,
                "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "min_p": 0.0, "presence_penalty": 1.5}
    # gpt-oss-20b — use the same generation params that produced coherent output
    # in the reasoning sweep (temp=0.7, top_p=0.9, top_k=20). The "authoritative"
    # recommendation of temp=1.0/top_p=1.0/top_k=0 removes all token filtering;
    # with local model.generate() this causes cascading garbage output.
    # Reasoning effort is controlled via system_suffix in the steering config.
    return {**base, "enable_thinking": False,
            "temperature": 0.7, "top_p": 0.9, "top_k": 20,
            "disable_quantization": True}


def write_config(model_id: str, model_tag: str, condition: str,
                 n: int, personas: list[str]) -> None:
    suffix = f"_{model_tag}" if model_tag else ""
    run_id = f"gbs_exact_replication_{condition}_{n}p{suffix}"

    env_kwargs: dict = {
        "num_rounds": 30,
        "low": 0,
        "feedback": "directional",
        "hide_group_size": True,
        "persona_mode": condition,
    }
    if condition != "plain":
        env_kwargs["personas"] = personas   # full list; adapter samples N at runtime

    # gpt-oss-20b: pin reasoning effort to low via system_suffix so we isolate
    # the persona/ToM effect rather than confounding it with reasoning depth.
    # Qwen3: enable_thinking=False already handles this.
    if "gpt-oss" in model_id:
        steering = {"default": "prompt_injection",
                    "default_config": {"system_suffix": "\nReasoning: low"},
                    "per_agent": {}}
    else:
        steering = {"default": "noop", "per_agent": {}}

    cfg = {
        "run_id": run_id,
        "game": {
            "family": "symbolic",
            "id": "gbs_exact_replication",
            "env_kwargs": env_kwargs,
        },
        "episodes": 200,
        "model": model_cfg(model_id),
        "agents": {"count": n, "concurrency": "sequential", "max_parse_retries": 5},
        "steering": steering,
        "logging": {"dir": "logs/picking_sweep/"},
        "wandb": {
            "enabled": True,
            "project": WANDB_PROJECT,
            "name": run_id,
            "tags": ["picking", condition, f"{n}p", model_tag or "qwen3", "picking_sweep"],
        },
    }
    path = OUT / f"{run_id}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  wrote {path}")


def main() -> None:
    personas = load_personas()
    print(f"Loaded {len(personas)} personas from {PERSONAS_PATH}")

    models = [
        ("openai/gpt-oss-20b", "20b"),
        ("Qwen/Qwen3-14B",     "14b"),
    ]
    for model_id, model_tag in models:
        for condition in CONDITIONS:
            for n in PLAYERS:
                write_config(model_id, model_tag, condition, n, personas)

    total = len(models) * len(CONDITIONS) * len(PLAYERS)
    print(f"\nGenerated {total} configs -> {OUT}/")


if __name__ == "__main__":
    # NOTE: re-running regenerates all configs with hardcoded defaults.
    # If you have bumped episode counts or made per-file edits, check git diff first.
    main()
