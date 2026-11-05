"""Per agent and episode, what the agent tables (summarize_runs.py) do not keep, read from the episode JSONLs:

  why turns break     cut_turns: turns whose final answer was cut off at max_tokens (finish_reason 'length');
                      failed_cut: unparseable turns whose every attempt was cut off (vs a format error)
  length bins         turns and unparseable turns by the first attempt's generated tokens (bins BINS), so that
                      unparseable rates can be compared within game at matched length
  echo                for each other agent: the share of this agent's word 4-grams that the other agent wrote
                      earlier in the episode (mean over this agent's turns); JSON {other: share}

One row per (run_id, episode, agent): joins the agent tables on those keys. Only reads the logs.

    python scripts/turn_stats.py <log dir> [<fallback log dir>] --out results/<sweep>/turn_stats.csv.gz --jobs 10
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing
import re
from pathlib import Path

BINS = (128, 256, 512, 768, 1000)            # upper edges; the last bin is > 1000 tokens (max_tokens 1024)
N = 4                                        # n-gram length for echo
_WORD = re.compile(r"[a-z0-9']+")


def ngrams(text: str) -> set:
    w = _WORD.findall((text or "").lower())
    return {tuple(w[i:i + N]) for i in range(len(w) - N + 1)}


def bin_of(tokens) -> int:
    t = tokens or 0
    return next((i for i, edge in enumerate(BINS) if t <= edge), len(BINS))


def failed(rec) -> bool:
    return str(rec.get("parse_ok", True)).lower() == "false"


def episode_stats(path: str) -> list:
    """Rows of one episode JSONL (run_id from the run.json next to it)."""
    p = Path(path)
    run_id = json.loads((p.parent / "run.json").read_text(encoding="utf-8"))["run_id"]
    episode = int(p.name.split("_")[1].split(".")[0])
    agents, seen = {}, {}                    # aid -> stats; aid -> n-grams written so far
    with open(p, encoding="utf-8", errors="replace") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    for turn in sorted({r.get("turn", 0) for r in recs}):
        step = [r for r in recs if r.get("turn", 0) == turn and "agent_id" in r]
        for r in step:
            aid = r["agent_id"]
            a = agents.setdefault(aid, {"turns": 0, "failed": 0, "cut_turns": 0, "failed_cut": 0,
                                        "t_bins": [0] * (len(BINS) + 1), "f_bins": [0] * (len(BINS) + 1),
                                        "echo": {}, "echo_n": 0})
            attempts = r.get("attempts") or []
            first = attempts[0].get("n_tokens") if attempts else r.get("n_tokens")
            fail = failed(r)
            b = bin_of(first)
            a["turns"] += 1
            a["t_bins"][b] += 1
            a["cut_turns"] += r.get("finish_reason") == "length"
            if fail:
                a["failed"] += 1
                a["f_bins"][b] += 1
                reasons = [x.get("finish_reason") for x in attempts] or [r.get("finish_reason")]
                a["failed_cut"] += all(x == "length" for x in reasons)
            mine = ngrams(r.get("completion"))
            if mine:
                a["echo_n"] += 1
                for other, theirs in seen.items():
                    if other != aid:
                        a["echo"][other] = a["echo"].get(other, 0.0) + len(mine & theirs) / len(mine)
        for r in step:                       # simultaneous moves: an agent sees the others' text only afterwards
            seen.setdefault(r["agent_id"], set()).update(ngrams(r.get("completion")))
    rows = []
    for aid, a in sorted(agents.items()):
        row = {"run_id": run_id, "episode": episode, "agent": aid, "turns": a["turns"], "failed": a["failed"],
               "cut_turns": a["cut_turns"], "failed_cut": a["failed_cut"],
               "echo": json.dumps({o: round(v / a["echo_n"], 4) for o, v in sorted(a["echo"].items())}
                                  if a["echo_n"] else {}, separators=(",", ":"))}
        for i in range(len(BINS) + 1):
            row[f"t_b{i}"], row[f"f_b{i}"] = a["t_bins"][i], a["f_bins"][i]
        rows.append(row)
    return rows


def safe(path: str) -> list:
    try:
        return episode_stats(path)
    except Exception as e:                    # a half-written episode must not stop the pass
        print(f"skip {path}: {e}", flush=True)
        return []


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="log dir(s) of one sweep; the first wins for an episode logged twice")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    files = {}
    for logs in map(Path, a.logs):
        if logs.is_dir():
            for s in logs.rglob("episode_*.summary.json"):   # finished episodes only
                j = s.with_name(s.name[:-len(".summary.json")] + ".jsonl")
                if j.exists() and (j.parent / "run.json").exists():
                    files.setdefault(str(j.relative_to(logs)), str(j))
    paths = sorted(files.values())
    print(f"{len(paths)} finished episodes", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "episode", "agent", "turns", "failed", "cut_turns", "failed_cut", "echo"] + \
             [f"{k}{i}" for i in range(len(BINS) + 1) for k in ("t_b", "f_b")]
    n = 0
    with gzip.open(a.out, "wt", encoding="utf-8", newline="") as f, multiprocessing.Pool(a.jobs) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, rows in enumerate(pool.imap_unordered(safe, paths, chunksize=32), 1):
            w.writerows(rows)
            n += len(rows)
            if i % 5000 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)} episodes", flush=True)
    print(f"{n} agent-episode rows -> {a.out}")


if __name__ == "__main__":
    main()
