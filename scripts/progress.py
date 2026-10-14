"""How far a sweep's SLURM tasks have got, read from their logs (slurm-logs/games-*.out).

    python scripts/progress.py                            # every games-*.out
    python scripts/progress.py --configs config/runs/main # + each game's full total (queued tasks too)
    python scripts/progress.py 51880001 --tasks           # one array job, one line per task

Per task: the episodes it was given (its `... episode(s) to play in shard` line), how many are
done / failed, and how long ago it last logged (a long silence = stuck or dead). A resubmitted task
counts only what it was given this time; finished episodes from earlier submissions are in --configs.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

TO_PLAY = re.compile(r"\] \d+ run\(s\), (\d+) episode\(s\) to play in shard \d+/\d+(?: of (\S+))?")
DONE = re.compile(r"^\[[\d:]+\] (\S+) episode \d+ done after", re.M)
FAILED = re.compile(r"^\[[\d:]+\] ERROR (\S+) episode", re.M)
EPISODES = re.compile(r"^episodes: (\d+)$", re.M)


def task_progress(path: Path) -> dict:
    text = path.read_text(errors="replace")
    to_play = TO_PLAY.search(text)
    done, failed = DONE.findall(text), FAILED.findall(text)
    if to_play and to_play.group(2):
        group = Path(to_play.group(2)).name
    else:  # logs from before the group was printed: the group is the run id's prefix
        group = (done or failed or ["(starting)"])[0].split("__")[0]
    return {"task": path.stem.removeprefix("games-"), "group": group,
            "total": int(to_play.group(1)) if to_play else None, "done": len(done), "failed": len(failed),
            "idle_min": (time.time() - path.stat().st_mtime) / 60}


def config_totals(configs: Path) -> dict:
    """{group: runs x episodes} from the generated configs."""
    return {g.name: sum(int(EPISODES.search(f.read_text()).group(1)) for f in g.glob("*.yaml"))
            for g in sorted(configs.iterdir()) if g.is_dir()}


def bar(pct: float) -> str:
    return f"{'#' * int(pct / 5):<20} {pct:5.1f}%"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("job", nargs="?", default="", help="array job id (default: every games-*.out)")
    p.add_argument("--logs", default="slurm-logs")
    p.add_argument("--configs", help="generated sweep (config/runs/NAME): each game's full episode count")
    p.add_argument("--tasks", action="store_true", help="also print every task")
    args = p.parse_args(argv)

    tasks = [task_progress(f) for f in sorted(Path(args.logs).glob(f"games-{args.job}*.out"))]
    totals = config_totals(Path(args.configs)) if args.configs else {}
    if not tasks and not totals:
        print(f"no {args.logs}/games-{args.job}*.out")
        return 1
    if args.tasks:
        for t in tasks:
            pct = bar(100 * (t["done"] + t["failed"]) / t["total"]) if t["total"] else "not started"
            print(f"{t['task']:>16}  {t['group']:<26} {t['done']:>5} done {t['failed']:>4} failed of "
                  f"{t['total'] if t['total'] is not None else '?':>5}  {pct}   last log {t['idle_min']:.0f} min ago")
        print()

    groups = defaultdict(lambda: {"tasks": 0, "started": 0, "given": 0, "done": 0, "failed": 0, "idle": 1e9})
    for t in tasks:
        g = groups[t["group"]]
        g["tasks"] += 1
        g["started"] += t["total"] is not None
        g["given"] += t["total"] or 0
        g["done"] += t["done"]
        g["failed"] += t["failed"]
        g["idle"] = min(g["idle"], t["idle_min"])
    for name in totals:
        groups[name]  # games with no task log yet (all queued) still get a line
    of = "of sweep" if totals else "of started tasks"
    print(f"{'game group':<26} {'tasks':>6} {'done':>7} {'failed':>6} {of:>16}   progress")
    for name, g in sorted(groups.items()):
        total = totals.get(name, g["given"])
        pct = 100 * (g["done"] + g["failed"]) / total if total else 0.0
        print(f"{name:<26} {g['started']:>3}/{g['tasks']:<2} {g['done']:>7} {g['failed']:>6} {total:>16}   {bar(pct)}")
    done = sum(g["done"] for g in groups.values())
    failed = sum(g["failed"] for g in groups.values())
    total = sum(totals.values()) if totals else sum(g["given"] for g in groups.values())
    print(f"\nall: {done} done, {failed} failed of {total} episodes ({100 * (done + failed) / max(total, 1):.1f}%)"
          + ("" if totals else "  [only tasks that started; --configs config/runs/<sweep> for the whole sweep]"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
