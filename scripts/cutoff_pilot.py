"""Cut-off pilot: per token budget (one log dir each, config/sweeps/cutoff_pilot.yaml), how often answers still run
into the limit, how long they are, and what the budget costs. Reads only the episode JSONLs; corrupt lines are
skipped and counted.

Per budget x game group x condition kind (unsteered / steered at |c|):
  turns           final turns logged
  cut_first %     first attempt ended at max_tokens (finish_reason 'length')
  cut_final %     the turn's final answer still ended at max_tokens (after continuation / retries)
  unparse %       turns that fell back to the default move
  failed_cut %    unparseable turns whose every attempt was cut off (vs a format error), of all turns
  p50/p90/p99/max generated tokens of the first attempt
  tok_turn        mean generated tokens per turn, all attempts (the cost driver)
Reading: the smallest budget whose cut_first is near the unsteered level of the next larger budget, and whose
unparse no longer falls with more tokens, is enough; past it, more tokens only cost time.

    python scripts/cutoff_pilot.py <logs>/cutoff_pilot_5k <logs>/cutoff_pilot_10k <logs>/cutoff_pilot_15k \
        [--out results/cutoff_pilot.csv] [--jobs 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import re
import statistics
from collections import defaultdict
from pathlib import Path

RUN = re.compile(r"^(?P<group>[^_].*?_\d+p)__(?P<cond>[^_]+(?:_[a-z]+)*)(?:__(?P<trait>[a-z]+)_x(?P<c>-?[\d.]+))?"
                 r"(?:__obs-(?P<obs>\w+))?$")


def cell(run_id: str) -> tuple:
    m = RUN.match(run_id)
    if not m:
        return run_id, "?", "?"
    c = m.group("c")
    kind = "unsteered" if m.group("cond") == "none" else f"{m.group('cond')} |c|={abs(float(c)):g}"
    group = m.group("group") + ("" if not m.group("obs") else f" ({m.group('obs')})")
    return group, kind, m.group("trait") or ""


def episode(path: str) -> dict:
    """Counts of one episode file."""
    out = {"turns": 0, "cut_first": 0, "cut_final": 0, "unparse": 0, "failed_cut": 0, "bad_lines": 0,
           "first_tokens": [], "all_tokens": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                out["bad_lines"] += 1
                continue
            if "agent_id" not in r:
                continue
            att = r.get("attempts") or [{"finish_reason": r.get("finish_reason"), "n_tokens": r.get("n_tokens")}]
            out["turns"] += 1
            out["cut_first"] += att[0].get("finish_reason") == "length"
            out["cut_final"] += r.get("finish_reason") == "length"
            fail = str(r.get("parse_ok", True)).lower() == "false"
            out["unparse"] += fail
            out["failed_cut"] += fail and all(a.get("finish_reason") == "length" for a in att)
            out["first_tokens"].append(att[0].get("n_tokens") or 0)
            out["all_tokens"] += sum(a.get("n_tokens") or 0 for a in att)
    return out


def q(xs, p):
    if not xs:
        return ""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="one log dir per budget (<logging_dir>/cutoff_pilot_<k>k)")
    ap.add_argument("--out", default="results/cutoff_pilot.csv")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    jobs = []
    for d in a.logs:
        budget = Path(d).name.rsplit("_", 1)[-1]
        for run_dir in sorted(p for g in Path(d).iterdir() if g.is_dir() for p in g.iterdir() if p.is_dir()):
            for f in sorted(run_dir.glob("episode_*.jsonl")):
                jobs.append((budget, *cell(run_dir.name), str(f)))
    print(f"{len(jobs)} episode files in {len(a.logs)} budget(s)", flush=True)
    with multiprocessing.Pool(a.jobs) as pool:
        stats = pool.map(episode, [j[-1] for j in jobs], chunksize=8)
    agg = defaultdict(lambda: defaultdict(list))
    for (budget, group, kind, _trait, _f), s in zip(jobs, stats):
        for key in ((budget, group, kind), (budget, "ALL", kind), (budget, "ALL", "ALL")):
            for k, v in s.items():
                agg[key][k].append(v)
    rows = []
    for (budget, group, kind), s in sorted(agg.items()):
        turns = sum(s["turns"])
        if not turns:
            continue
        first = [t for ts in s["first_tokens"] for t in ts]
        pct = lambda k: round(100 * sum(s[k]) / turns, 2)
        rows.append({"budget": budget, "group": group, "kind": kind, "episodes": len(s["turns"]), "turns": turns,
                     "cut_first": pct("cut_first"), "cut_final": pct("cut_final"), "unparse": pct("unparse"),
                     "failed_cut": pct("failed_cut"), "p50": q(first, .5), "p90": q(first, .9),
                     "p99": q(first, .99), "max": max(first) if first else "",
                     "tok_turn": round(sum(s["all_tokens"]) / turns), "bad_lines": sum(s["bad_lines"]),
                     "mean_first": round(statistics.mean(first)) if first else ""})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["budget", "kind", "turns", "cut_first", "cut_final", "unparse", "failed_cut", "p50", "p90", "p99",
            "max", "tok_turn"]
    print("\nall games, by condition kind:")
    print("  ".join(f"{c:>10}" for c in cols))
    for r in rows:
        if r["group"] == "ALL":
            print("  ".join(f"{str(r[c]):>10}" for c in cols))
    print("\nper game (unsteered and ALL):")
    for r in rows:
        if r["group"] != "ALL" and r["kind"] == "unsteered":
            print(f"{r['budget']:>5} {r['group']:<28} cut_first {r['cut_first']:>6}%  unparse {r['unparse']:>6}%  "
                  f"p90 {r['p90']:>6}  max {r['max']:>6}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
