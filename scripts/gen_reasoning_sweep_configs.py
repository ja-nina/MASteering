"""Generate configs for the reasoning mode sweep.

Structure
---------
Tasks 0-17  : Qwen/Qwen3-4B  — 3 modes x 2 games x 3 player counts
Tasks 18-29 : gpt-oss-20b    — 2 modes (thinking + non_thinking) x 2 games x 3 players
              (noop is model-agnostic, so 4B baseline suffices)

Total: 30 configs → config/reasoning_sweep/

Modes
-----
noop         : no reasoning instruction, no thinking tokens   (pure baseline, 4B only)
non_thinking : explicit step-by-step reasoning prompt,        (prompted reasoning)
               no thinking tokens
thinking     : native thinking tokens (enable_thinking=True)

Usage
-----
python scripts/gen_reasoning_sweep_configs.py
"""
from pathlib import Path
import yaml

OUT = Path("config/reasoning_sweep")
OUT.mkdir(parents=True, exist_ok=True)

GAMES  = ["gbs", "beauty_contest"]
PLAYERS = [2, 3, 4]
MODES  = ["noop", "non_thinking", "thinking"]

WANDB_PROJECT = "ma-steering-tom-effectiveness"
GAME_ROUNDS   = {"beauty_contest": 10, "gbs": 20}

REASONING_PREFIX = (
    "Before answering, think step by step: "
    "what does the history tell you about the next best move? "
    "What are other players likely to do? "
    "What do they think I am likely to do, judging from their perspective?"
)

# ── per-model, per-mode settings ──────────────────────────────────────────────

def model_cfg(model_id: str, mode: str) -> dict:
    base = {"backend": "transformers", "model_id": model_id}
    if mode == "thinking":
        return {**base, "enable_thinking": True,
                "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    else:  # noop / non_thinking
        return {**base, "enable_thinking": False,
                "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                "repetition_penalty": 1.05}


def steering_cfg(mode: str) -> dict:
    if mode == "non_thinking":
        return {"default": "prompt_injection",
                "default_config": {"user_prefix": REASONING_PREFIX},
                "per_agent": {}}
    return {"default": "noop", "per_agent": {}}


def write_config(model_id: str, model_tag: str, mode: str, game: str, n: int) -> None:
    suffix = f"_{model_tag}" if model_tag else ""
    run_id = f"{game}_{mode}_{n}p{suffix}"
    cfg = {
        "run_id": run_id,
        "game": {
            "family": "symbolic",
            "id": game,
            "env_kwargs": {"num_rounds": GAME_ROUNDS[game]},
        },
        "episodes": 10,
        "model": model_cfg(model_id, mode),
        "agents": {"count": n, "concurrency": "sequential", "max_parse_retries": 5},
        "steering": steering_cfg(mode),
        "logging": {"dir": "logs/reasoning_sweep/"},
        "wandb": {
            "enabled": True,
            "project": WANDB_PROJECT,
            "name": run_id,
            "tags": [game, mode, f"{n}p", model_tag or "4b", "reasoning_sweep"],
        },
    }
    path = OUT / f"{run_id}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  wrote {path}")


# ── tasks 0-17: Qwen3-4B, all 3 modes ────────────────────────────────────────
for game in GAMES:
    for n in PLAYERS:
        for mode in MODES:
            write_config("Qwen/Qwen3-4B", "", mode, game, n)

# ── tasks 18-29: gpt-oss-20b, thinking + non_thinking only ───────────────────
for mode in ["thinking", "non_thinking"]:
    for game in GAMES:
        for n in PLAYERS:
            write_config("gpt-oss-20b", "20b", mode, game, n)

total = len(GAMES) * len(PLAYERS) * len(MODES) + 2 * len(GAMES) * len(PLAYERS)
print(f"\nGenerated {total} configs → {OUT}/")
