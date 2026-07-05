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
thinking     : native thinking tokens (enable_thinking=True) AND the explicit
               step-by-step reasoning prompt — two confounded changes at once.
noop_thinking: ablates that confound. Same reasoning-enabling mechanism as
               `thinking` (enable_thinking=True for Qwen3 / "Reasoning: high"
               for gpt-oss-20b) but the *prompt* is left exactly as `noop`
               (no step-by-step prefix). Isolates "does letting the model
               think help" from "does telling it to think step by step help".
               Only generated for gpt-oss-20b: for Qwen3, `thinking` already
               has no prompt prefix (steering=noop), so `noop_thinking` would
               be byte-for-byte identical to the existing `thinking` config —
               generating it would just duplicate compute for no new signal.
               (Exception: it IS generated for Qwen3 in the GBS tom sweep,
               below, to keep that grid symmetric across models.)
tom          : GBS-only. Same "no reasoning effort override" baseline as
               `noop`, but the prompt is the hand-written Theory-of-Mind
               prefix (TOM_PREFIX, reused verbatim from
               config/gbs_exp_prompt_all_tom.yaml, typos and all — kept
               as-is for continuity with that earlier experiment).
tom_thinking : GBS-only. TOM_PREFIX + the same reasoning-enabling mechanism
               as `thinking`/`noop_thinking`. Together `tom`/`tom_thinking`
               and `noop`/`noop_thinking` form a 2x2 (ToM prompt x thinking)
               ablation, mirrored across both Qwen3-4B and gpt-oss-20b.

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

# Reused verbatim (including typos) from config/gbs_exp_prompt_all_tom.yaml
# for continuity with that earlier all-agent ToM experiment.
TOM_PREFIX = (
    "Assume the other players are rational and are also attempting to model "
    "your reasoning.\nSelect the contribution that is most likely to be part "
    "of a mutually consistent set of choices across all players. Be concise, "
    "do not repeat thinking patterns and end your contribution, start your "
    "anwser with first explaining yyour reson STRATEGY: with "
    "CONTRIBUTION: <integer>.\n"
)

# ── per-model, per-mode settings ──────────────────────────────────────────────

def model_cfg(model_id: str, mode: str) -> dict:
    base = {"backend": "transformers", "model_id": model_id}
    is_qwen3 = "Qwen3" in model_id
    if is_qwen3:
        if mode in ("thinking", "noop_thinking", "tom_thinking"):
            # Qwen3 recommended thinking params
            return {**base, "enable_thinking": True,
                    "temperature": 0.6, "top_p": 0.95, "top_k": 20}
        else:
            # Qwen3 recommended non-thinking params
            return {**base, "enable_thinking": False,
                    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
                    "repetition_penalty": 1.05}
    else:
        # gpt-oss-20b: reasoning level via system_suffix; disable built-in MXFP4
        # quantization (requires torch>=2.4) and load in float16 instead
        return {**base, "enable_thinking": False,
                "temperature": 0.7, "top_p": 0.9, "top_k": 20,
                "disable_quantization": True}


def steering_cfg(model_id: str, mode: str) -> dict:
    is_qwen3 = "Qwen3" in model_id
    if is_qwen3:
        if mode == "non_thinking":
            return {"default": "prompt_injection",
                    "default_config": {"user_prefix": REASONING_PREFIX},
                    "per_agent": {}}
        if mode in ("tom", "tom_thinking"):
            return {"default": "prompt_injection",
                    "default_config": {"user_prefix": TOM_PREFIX},
                    "per_agent": {}}
        # noop, noop_thinking, thinking — none of these touch the prompt
        return {"default": "noop", "per_agent": {}}
    else:
        # gpt-oss-20b: steer reasoning depth via system_suffix
        if mode == "thinking":
            return {"default": "prompt_injection",
                    "default_config": {
                        "system_suffix": "\nReasoning: high",
                        "user_prefix": REASONING_PREFIX,
                    },
                    "per_agent": {}}
        elif mode == "noop_thinking":
            # same reasoning effort as `thinking`, but no prompt change
            return {"default": "prompt_injection",
                    "default_config": {
                        "system_suffix": "\nReasoning: high",
                    },
                    "per_agent": {}}
        elif mode == "non_thinking":
            return {"default": "prompt_injection",
                    "default_config": {
                        "system_suffix": "\nReasoning: medium",
                        "user_prefix": REASONING_PREFIX,
                    },
                    "per_agent": {}}
        elif mode == "tom":
            # same "no effort override" baseline as noop, ToM prompt instead
            return {"default": "prompt_injection",
                    "default_config": {"user_prefix": TOM_PREFIX},
                    "per_agent": {}}
        elif mode == "tom_thinking":
            return {"default": "prompt_injection",
                    "default_config": {
                        "system_suffix": "\nReasoning: high",
                        "user_prefix": TOM_PREFIX,
                    },
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
        "steering": steering_cfg(model_id, mode),
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


def main() -> None:
    # ── tasks 0-17: Qwen3-4B, all 3 modes ────────────────────────────────────
    # (noop_thinking deliberately excluded here — see module docstring)
    for game in GAMES:
        for n in PLAYERS:
            for mode in MODES:
                write_config("Qwen/Qwen3-4B", "", mode, game, n)

    # ── gpt-oss-20b: noop, non_thinking, thinking, noop_thinking ─────────────
    modes_20b = ["thinking", "noop", "non_thinking", "noop_thinking"]
    for mode in modes_20b:
        for game in GAMES:
            for n in PLAYERS:
                write_config("openai/gpt-oss-20b", "20b", mode, game, n)

    total = (len(GAMES) * len(PLAYERS) * len(MODES)
             + len(modes_20b) * len(GAMES) * len(PLAYERS))
    print(f"\nGenerated {total} configs → {OUT}/")


if __name__ == "__main__":
    # NOTE: re-running this regenerates every file below with hardcoded
    # defaults (episodes=10, etc.), clobbering any manual per-file overrides
    # (e.g. bumped episode counts). Check `git diff` before running against
    # an existing config/reasoning_sweep/ directory with in-flight edits.
    main()
