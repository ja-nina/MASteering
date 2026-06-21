"""Generate 18 YAML configs for the reasoning mode sweep.

3 modes x 2 games x 3 player counts = 18 configs → config/reasoning_sweep/

Modes
-----
noop         : no reasoning instruction, no thinking tokens  (pure baseline)
non_thinking : explicit step-by-step reasoning prompt,       (prompted reasoning)
               no thinking tokens
thinking     : Qwen3 native thinking tokens (enable_thinking=True)

Usage
-----
python scripts/gen_reasoning_sweep_configs.py
"""
from pathlib import Path
import yaml

OUT = Path("config/reasoning_sweep")
OUT.mkdir(parents=True, exist_ok=True)

GAMES = ["beauty_contest", "gbs"]
PLAYERS = [2, 3, 4]
MODES = ["noop", "non_thinking", "thinking"]

WANDB_PROJECT = "ma-steering-tom-effectiveness"

GAME_ROUNDS = {"beauty_contest": 10, "gbs": 20}

REASONING_PREFIX = (
    "Assume the other players are rational and are also attempting to model your reasoning."
    "Before answering, think step by step: "
    "what does the history tell you about the next best move? "
    "What are other players likely to do? What do they think I am likely to do, judging from their perspective?"
)

for game in GAMES:
    for n in PLAYERS:
        for mode in MODES:
            run_id = f"{game}_{mode}_{n}p"

            cfg: dict = {
                "run_id": run_id,
                "game": {
                    "family": "symbolic",
                    "id": game,
                    "env_kwargs": {
                        "num_rounds": GAME_ROUNDS[game],
                        "num_players": n,
                    },
                },
                "episodes": 10,
                "model": {
                    "backend": "transformers",
                    "model_id": "Qwen/Qwen3-4B",
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "enable_thinking": mode == "thinking",
                },
                "agents": {
                    "count": n,
                    "concurrency": "sequential",
                    "max_parse_retries": 5,
                },
                "logging": {
                    "dir": f"logs/reasoning_sweep/",
                },
                "wandb": {
                    "enabled": True,
                    "project": WANDB_PROJECT,
                    "name": run_id,
                    "tags": [game, mode, f"{n}p", "reasoning_sweep"],
                },
            }

            if mode == "non_thinking":
                cfg["steering"] = {
                    "default": "prompt_injection",
                    "default_config": {"user_prefix": REASONING_PREFIX},
                    "per_agent": {},
                }
            else:
                cfg["steering"] = {
                    "default": "noop",
                    "per_agent": {},
                }

            path = OUT / f"{run_id}.yaml"
            with open(path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"  wrote {path}")

print(f"\nGenerated {len(GAMES) * len(PLAYERS) * len(MODES)} configs → {OUT}/")
