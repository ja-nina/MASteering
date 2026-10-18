"""Episode scores for episodes logged before summaries had `total_rewards`.

    python scripts/backfill_rewards.py logs/main [fallback/main] [--jobs 8]

Their summary's `final_rewards` is only the last step's rewards: the episode's score in games that pay once at
the end, but not in the games in PER_STEP_GAMES, which pay every step. For those, the score is rebuilt and written
to episode_N.rewards.json next to the summary ({"total_rewards": {agent: score}, "source": ...}); the summary and
JSONL are left as they are. scripts/summarize_runs.py takes the reward from the summary's total_rewards, else from
this file, else final_rewards. Episodes that already have either are skipped, so it can be rerun.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# game id -> where its episode score is, for the games whose steps pay out (every other game pays once, at the end)
PER_STEP_GAMES = {
    "harvest": "final_info.totals",      # the round's take each step; the running totals are in the step info
    "overcooked": "final_info.score",    # the score change each step; the team's score is in the step info
    "beauty_contest": "jsonl",           # the round's win share each step: summed over the JSONL's turns
}


def side_path(summary_path: Path) -> Path:
    return summary_path.with_name(summary_path.name[:-len(".summary.json")] + ".rewards.json")


def rebuild(game: str, summary: dict, summary_path: Path) -> dict:
    """{agent: episode score} of an episode of a PER_STEP_GAMES game."""
    source, info = PER_STEP_GAMES[game], summary.get("final_info") or {}
    agents = summary.get("final_rewards") or {}
    if source == "final_info.totals":
        return {a: float(v) for a, v in info["totals"].items()}
    if source == "final_info.score":  # shared by the team
        return {a: float(info["score"]) for a in agents}
    totals = {a: 0.0 for a in agents}
    jsonl = summary_path.with_name(summary_path.name[:-len(".summary.json")] + ".jsonl")
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            totals[rec["agent_id"]] = totals.get(rec["agent_id"], 0.0) + float(rec["reward"])
    return totals


def backfill(job) -> str:
    """(game, summary path) -> 'written' | 'skipped' | 'failed: <why>'."""
    game, spath = job
    spath = Path(spath)
    side = side_path(spath)
    if side.exists():
        return "skipped"
    summary = json.loads(spath.read_text(encoding="utf-8"))
    if "total_rewards" in summary:
        return "skipped"
    try:
        out = {"total_rewards": rebuild(game, summary, spath), "source": PER_STEP_GAMES[game]}
    except (KeyError, OSError, ValueError) as e:
        return f"failed: {spath} ({type(e).__name__}: {e})"
    tmp = side.with_name(f"{side.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    tmp.replace(side)
    return "written"


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="log dir(s) of one sweep (every copy of an episode gets its file)")
    p.add_argument("--jobs", type=int, default=8, help="processes reading episodes in parallel")
    args = p.parse_args(argv)
    jobs = []
    for logs in map(Path, args.logs):
        for run_json in sorted(logs.rglob("run.json")) if logs.is_dir() else []:
            game = json.loads(run_json.read_text(encoding="utf-8"))["game"]["id"]
            if game in PER_STEP_GAMES:
                jobs += [(game, str(s)) for s in sorted(run_json.parent.glob("episode_*.summary.json"))]
    print(f"{len(jobs)} episodes of {', '.join(sorted(PER_STEP_GAMES))}", flush=True)
    counts, failed = {"written": 0, "skipped": 0}, []
    with multiprocessing.Pool(args.jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(backfill, jobs, chunksize=16), 1):
            if result.startswith("failed"):
                failed.append(result)
            else:
                counts[result] += 1
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    for f in failed[:20]:
        print(f"  {f}")
    print(f"{counts['written']} written, {counts['skipped']} skipped (already had their score), {len(failed)} failed")


if __name__ == "__main__":
    main()
