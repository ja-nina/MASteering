"""Token counts for episodes logged before the runner recorded them.

    python scripts/backfill_tokens.py logs/main [fallback/main] [--jobs 8]

Per agent of an episode: the tokens the model read (every attempt's prompt: the templated text and its images;
retries re-read the prompt with the parser's feedback) and generated (every attempt's answer). Written to
episode_N.tokens.json next to the summary ({agent: {"in": n, "out": n}}); scripts/summarize_runs.py takes the
counts from the summary's agent_stats, else from this file. Read tokens are counted the way the engine counts
them: the model's tokenizer on the logged model input, plus ceil(w/p) * ceil(h/p) + 2 per image, with the image
size redrawn from the run's config (the main sweep saves no images). Episodes that already have counts are
skipped, so it can be rerun.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def side_path(summary_path: Path) -> Path:
    return summary_path.with_name(summary_path.name[:-len(".summary.json")] + ".tokens.json")


def count_tokens(jsonl: Path, encode, per_image) -> dict:
    """{agent: {"in": tokens read, "out": tokens generated}} of one episode. encode(text) -> token count;
    per_image: tokens of one image of this run (None when unknown: then image turns leave "in" unknown)."""
    out = {}
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            a = out.setdefault(rec["agent_id"], {"in": 0, "out": 0})
            attempts = rec.get("attempts") or [{"user_prompt": rec.get("user_prompt", ""), "n_tokens": rec.get("n_tokens", 0)}]
            images = rec.get("images") or 0
            n_images = images if isinstance(images, int) else len(images)
            final_user = rec.get("user_prompt", "")
            if rec.get("model_input"):  # the exact templated prompt of the final attempt
                around = encode(rec["model_input"]) - encode(final_user)
            else:
                around = encode(rec.get("system_prompt", ""))
            img = None if n_images and per_image is None else n_images * (per_image or 0)
            for att in attempts:
                a["out"] += att.get("n_tokens") or 0
                if a["in"] is not None:
                    a["in"] = None if img is None else a["in"] + around + encode(att.get("user_prompt") or "") + img
    return out


def image_tokens(run_dir: Path, pixels_per_token: int):
    """Tokens of one image of this run: its first observation redrawn from the config and episode 0's seed."""
    from testbed.config import RunConfig
    from testbed.registry import build_game
    from testbed.visual import fit, observe
    cfg = RunConfig.from_dict(json.loads((run_dir / "run.json").read_text(encoding="utf-8")))
    if cfg.observation == "text":
        return 0
    env, renderer, _ = build_game(family=cfg.game_family, game_id=cfg.game_id, num_players=cfg.players,
                                  env_kwargs={"seed": cfg.episode_seed(0), **cfg.env_kwargs})
    env.reset()
    try:
        aid, obs = env.pending()[0]
        _, images = observe(renderer, cfg.observation, obs, aid, getattr(env, "context", None))
    finally:
        getattr(env, "close", lambda: None)()
    if not images:
        return 0
    im = fit(images[0], cfg.image_size)
    p = pixels_per_token
    return math.ceil(im.width / p) * math.ceil(im.height / p) + 2


_TOKENIZER = None


def _work(job):
    """(summary path, tokens per image, hf repo) -> whether a side file was written."""
    global _TOKENIZER
    spath, per_image, repo = Path(job[0]), job[1], job[2]
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(repo)
    jsonl = spath.with_name(spath.name[:-len(".summary.json")] + ".jsonl")
    if not jsonl.exists():
        return False
    counts = count_tokens(jsonl, lambda text: len(_TOKENIZER.encode(text, add_special_tokens=False)), per_image)
    side = side_path(spath)
    tmp = side.with_name(side.name + ".tmp")
    tmp.write_text(json.dumps(counts), encoding="utf-8")
    tmp.replace(side)
    return True


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args(argv)
    from persvecfast.models import MODELS
    from testbed.mia_engine import _pixels_per_token
    jobs, per_run, pixels = [], {}, {}
    for logs in map(Path, args.logs):
        if not logs.is_dir():
            continue
        for spath in sorted(logs.rglob("episode_*.summary.json")):
            if side_path(spath).exists():
                continue
            summary = json.loads(spath.read_text(encoding="utf-8"))
            if all(a.get("tokens_in") is not None for a in (summary.get("agent_stats") or {"": {}}).values()):
                continue  # counted while it ran
            run_dir = spath.parent
            if run_dir not in per_run:
                run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                spec = MODELS[run["model"]]
                if spec.key not in pixels:
                    pixels[spec.key] = _pixels_per_token(spec)
                try:
                    per_run[run_dir] = (image_tokens(run_dir, pixels[spec.key]), spec.hf_repo)
                except Exception as e:  # noqa: BLE001 - its image turns stay uncounted, text turns still count
                    print(f"{run_dir.name}: image size unknown ({type(e).__name__}: {e})")
                    per_run[run_dir] = (None, spec.hf_repo)
            jobs.append((str(spath), *per_run[run_dir]))
    print(f"{len(jobs)} episodes to count")
    with multiprocessing.Pool(args.jobs) as pool:
        done = sum(pool.imap_unordered(_work, jobs, chunksize=64))
    print(f"wrote {done} token files")


if __name__ == "__main__":
    main()
