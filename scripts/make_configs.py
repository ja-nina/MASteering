"""Expand a sweep file into one run config per (game, players, condition, trait, coefficient).

    python scripts/make_configs.py config/sweeps/main.yaml
    -> config/runs/main/<game>_<players>p/<run_id>.yaml
    python scripts/make_configs.py config/sweeps/main.yaml --model qwen3.5-2b
    -> config/runs/main_qwen3.5-2b/...   (another PersVecFast model; its own runs and logs)

Each <game>_<players>p directory is meant to be one SLURM array (scripts/slurm/submit_sweep.sh):
its runs share an engine pair and play concurrently. A condition is a steering-group template
(pool + selection, see testbed/steering.py) that gets each trait and coefficient filled in;
`null` is the unsteered baseline (written once per game). Prompt-only conditions (`via: prompt`)
get one run per coefficient sign, as a persona prompt has no strength. Conditions with `team` or `roles`
are only generated for games marked `roles: true`. A game's `observation` (one mode or a list of
text | image | both) multiplies its runs; non-text runs get an `__obs-<mode>` suffix.
Logs go to <logging_dir>/<name>/<group>; a sweep with `log_name: <other sweep>` logs into that sweep's dirs
instead (its configs stay under config/runs/<name>): same run_ids, so the other sweep's finished episodes count
as done and a larger `episodes` only adds the missing ones; summarize/report on the other sweep's logs see both.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

RUN_KEYS = ("model", "vectors_dir", "traits_dir", "episodes", "max_tokens", "max_parse_retries", "max_turns",
            "monitor", "engine", "logging_dir", "image_size", "save_images")


def _merged(base: dict, over: dict) -> dict:
    """base with over's keys replaced, nested dicts merged key by key."""
    return {**base, **{k: _merged(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
                       for k, v in over.items()}}


def load_sweep(path, model=None) -> dict:
    """The sweep file, with `model` swapped for another PersVecFast model key if given; a different
    model gets its own name (<name>_<model>), so its configs and logs never mix with the sweep's own.
    The sweep's `models: {<key>: {...}}` overrides run settings for that model only (e.g. `engine` for a
    model that needs 2 GPUs per engine), and may narrow its grid (`coefficients`, `traits`: a subset keeps the
    run_ids of the cells it keeps); other models' configs do not change."""
    sweep = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if model and model != sweep["model"]:
        sweep = {**sweep, "model": model, "name": f"{sweep['name']}_{model}",
                 **({"log_name": f"{sweep['log_name']}_{model}"} if sweep.get("log_name") else {})}
    over = (sweep.get("models") or {}).get(sweep["model"]) or {}
    unknown = set(over) - set(RUN_KEYS) - {"coefficients", "traits"}
    if unknown:
        raise ValueError(f"models.{sweep['model']}: only run settings, coefficients and traits can be overridden, "
                         f"not {sorted(unknown)}")
    return _merged(sweep, over)


def expand(sweep: dict) -> dict:
    """{relative path: run config} for every cell of the sweep."""
    base = {k: sweep[k] for k in RUN_KEYS if k in sweep}
    runs = {}
    for game_name, game in sweep["games"].items():
        players = game.get("players", [2])
        for n in players if isinstance(players, list) else [players]:
            group_dir = f"{game_name}_{n}p"
            game_cfg = {"family": game["family"], "id": game["id"], "players": n,
                        "env_kwargs": game.get("env_kwargs") or {}}
            observations = game.get("observation", "text")
            for obs, (cond_name, template) in ((o, c) for o in (observations if isinstance(observations, list)
                                                                else [observations])
                                               for c in sweep["conditions"].items()):
                if template and ("team" in template or "roles" in template) and not game.get("roles"):
                    continue
                coeffs = sweep["coefficients"]
                if template and template.get("via") == "prompt":  # a persona prompt has no strength,
                    coeffs = sorted({1 if c >= 0 else -1 for c in coeffs})  # only a sign: one run per sign
                cells = [(None, None)] if template is None else \
                    [(t, c) for t in sweep["traits"] for c in coeffs]
                for trait, coeff in cells:
                    run_id = f"{group_dir}__{cond_name}" + (f"__{trait}_x{coeff:g}" if trait else "") +                         ("" if obs == "text" else f"__obs-{obs}")
                    steering = None if template is None else {**template, "trait": trait, "coefficient": coeff}
                    logging_dir = (Path(base.get("logging_dir", "logs")) / sweep.get("log_name", sweep["name"])
                                   / group_dir).as_posix()
                    runs[f"{group_dir}/{run_id}.yaml"] = {"run_id": run_id, **base, "logging_dir": logging_dir,
                                                          "game": game_cfg, "observation": obs,
                                                          "steering": steering}
    return runs


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweep")
    p.add_argument("--out", default="config/runs")
    p.add_argument("--model", help="run the sweep with another PersVecFast model key, e.g. qwen3.5-2b")
    p.add_argument("--print-name", action="store_true", help="only print the sweep's (model-suffixed) name")
    args = p.parse_args(argv)
    sweep = load_sweep(args.sweep, args.model)
    if args.print_name:
        print(sweep["name"])
        return
    out = Path(args.out) / sweep["name"]
    if out.exists():
        shutil.rmtree(out)  # generated files only: a sweep edit must not leave stale runs behind
    runs = expand(sweep)
    for rel, cfg in runs.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    groups = sorted({rel.split("/")[0] for rel in runs})
    print(f"{len(runs)} run configs in {len(groups)} groups under {out}")


if __name__ == "__main__":
    main()
