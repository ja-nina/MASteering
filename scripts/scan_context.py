"""Which runs hit the context limit, and how much context their episodes need.

For every run dir under the given log folders: turns, turns whose prompt alone filled the context
(finish_reason prompt_too_long: nothing generated, the parser fell back), episodes with at least one such turn,
and the longest prompt seen (text tokens of the chat-templated model input; image tokens come on top).

    python scripts/scan_context.py $L/main $L/main_qwen3.5-2b $L/main_qwen3.8-27b --out context_scan.csv
    python scripts/scan_context.py $L/main_qwen3.5-2b/poker_4p --tokenizer Qwen/Qwen3.5-2B

Runs are scanned in parallel (--workers); the per-game summary at the end lists only games with any such turn.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

NEEDLE = '"prompt_too_long"'


def scan_run(run_dir: str) -> dict:
    turns = too_long = 0
    eps_hit = 0
    longest = ""
    for f in sorted(Path(run_dir).glob("episode_*.jsonl")):
        hit = False
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"parse_ok"' not in line:
                    continue  # the summary line
                turns += 1
                if NEEDLE in line:
                    too_long += 1
                    hit = True
                # the longest model input: only parse lines that could beat the current one
                if len(line) > len(longest) + 1000:
                    try:
                        mi = json.loads(line).get("model_input") or ""
                    except json.JSONDecodeError:
                        continue
                    if len(mi) > len(longest):
                        longest = mi
        eps_hit += hit
    p = Path(run_dir)
    return {"sweep": p.parent.parent.name, "game": p.parent.name, "run_id": p.name, "turns": turns,
            "too_long_turns": too_long, "episodes_hit": eps_hit,
            "episodes": len(list(p.glob("episode_*.jsonl"))), "longest_chars": len(longest), "_text": longest}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="log folders: a sweep, a game, or a run dir")
    ap.add_argument("--out", default="context_scan.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer for exact token counts (default: from the sweep name: 2b/27b/else 9B)")
    a = ap.parse_args(argv)
    runs = sorted({str(p.parent) for r in a.roots for p in Path(r).rglob("episode_*.jsonl")})
    if not runs:
        raise SystemExit("no episode_*.jsonl found")
    print(f"scanning {len(runs)} runs with {a.workers} workers ...", file=sys.stderr)
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(scan_run, runs, chunksize=4))

    from transformers import AutoTokenizer  # only the longest prompt per run is tokenized
    toks = {}

    def tokenizer_for(sweep: str):
        name = a.tokenizer or ("Qwen/Qwen3.5-2B" if "2b" in sweep else
                               "Qwen/Qwen3.8-27B" if "27b" in sweep else "Qwen/Qwen3.5-9B")
        if name not in toks:
            toks[name] = AutoTokenizer.from_pretrained(name)
        return toks[name]

    for r in rows:
        text = r.pop("_text")
        r["longest_tokens"] = len(tokenizer_for(r["sweep"])(text).input_ids) if text else 0
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    games = defaultdict(lambda: [0, 0, 0, 0, 0, 0])  # runs, runs hit, turns, too-long turns, episodes hit, max tokens
    for r in rows:
        g = games[(r["sweep"], r["game"])]
        g[0] += 1
        g[1] += r["too_long_turns"] > 0
        g[2] += r["turns"]
        g[3] += r["too_long_turns"]
        g[4] += r["episodes_hit"]
        g[5] = max(g[5], r["longest_tokens"])
    print(f"{'sweep':22s} {'game':28s} {'runs hit':>12s} {'too-long turns':>18s} {'eps hit':>8s} {'max prompt tok':>15s}")
    for (sweep, game), (n, hit, t, tl, eh, mx) in sorted(games.items()):
        if tl:
            print(f"{sweep:22s} {game:28s} {hit:5d}/{n:<6d} {tl:8d}/{t:<9d} {eh:8d} {mx:15d}")
    print(f"\nall runs: {a.out}  (games with no too-long turn are not listed above)")


if __name__ == "__main__":
    main()
