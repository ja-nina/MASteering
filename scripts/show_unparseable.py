"""Print examples of unparseable turns (parse_ok false: every attempt failed, the runner took the fallback move).

Reads episode JSONL under one or more folders (a run dir, a game dir, or a whole log root) and prints, per turn:
the run (game, steering, observation), the agent, every attempt's answer and the parser's error.

    python scripts/show_unparseable.py logs/smoke                       # 5 random examples
    python scripts/show_unparseable.py <log_root>/gbs_3p -n 10 --match obs-image
    python scripts/show_unparseable.py <log_root> --match sycophantic_x-2 --chars 1500
    python scripts/show_unparseable.py <log_root> --count               # unparseable / turns per run, share cut off
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def as_obj(v):
    """Fields may be logged as strings of Python reprs; turn them back into objects when possible."""
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def failed(rec) -> bool:
    return str(rec.get("parse_ok")).lower() == "false"


def clip(s, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else f"{s[:n // 2]}\n   [... {len(s) - n} chars cut ...]\n{s[-n // 2:]}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("-n", type=int, default=5, help="examples to print")
    ap.add_argument("--match", default="", help="only files whose path contains this (e.g. obs-image, evil_x2)")
    ap.add_argument("--chars", type=int, default=800, help="max characters per answer shown")
    ap.add_argument("--prompt", action="store_true", help="also print the last user prompt")
    ap.add_argument("--count", action="store_true", help="only counts per run")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    files = sorted(p for r in a.roots for p in Path(r).rglob("episode_*.jsonl") if a.match in str(p))
    if not files:
        raise SystemExit("no episode_*.jsonl found")
    random.seed(a.seed)
    random.shuffle(files)
    hits, counts = [], {}
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "parse_ok" not in rec:
                    continue  # the summary line
                c = counts.setdefault(rec.get("run_id", f.parent.name), [0, 0, 0, 0])
                c[1] += 1
                if failed(rec):
                    c[0] += 1
                    c[2] += rec.get("finish_reason") == "length"
                    c[3] += rec.get("finish_reason") == "prompt_too_long"
                    if not a.count and len(hits) < a.n:
                        hits.append((f, rec))
        if not a.count and len(hits) >= a.n:
            break
    if a.count:
        print("unparse%   bad/turns   cut off%  too long%  run   (of the unparseable turns: cut off = answer hit the "
              "token limit; too long = the prompt alone filled the context, nothing was generated)")
        for run, (bad, tot, cut, long_) in sorted(counts.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
            pct = (lambda k: 100 * k / bad if bad else 0)
            print(f"{100 * bad / tot:6.1f}%  {bad:5d}/{tot:<6d} {pct(cut):6.0f}%   {pct(long_):6.0f}%    {run}")
        return
    if not hits:
        print(f"no unparseable turns in {len(files)} files")
    for f, rec in hits:
        steer = as_obj(rec.get("steering"))
        label = steer.get("label") if isinstance(steer, dict) else "unsteered"
        print("=" * 100)
        print(f"{rec.get('run_id')}  episode {rec.get('episode')}  turn {rec.get('turn')}  {rec.get('agent_id')}  "
              f"[{label}]  finish={rec.get('finish_reason')}  tokens={rec.get('n_tokens')}  "
              f"fallback move={rec.get('parsed_action')!r}")
        print(f"file: {f}")
        if a.prompt:
            print("--- last user prompt:\n" + clip(rec.get("user_prompt"), a.chars))
        attempts = as_obj(rec.get("attempts")) or [{"completion": rec.get("completion")}]
        for att in attempts if isinstance(attempts, list) else []:
            print(f"--- attempt {att.get('attempt', '?')}: answer")
            print(clip(att.get("completion"), a.chars))
            err = att.get("parse_error") or att.get("error")
            if err:
                print(f"    parser: {clip(err, 300)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
