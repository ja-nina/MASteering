"""How far a sweep's SLURM tasks have got, read from their logs (slurm-logs/games-*.out).

    python scripts/progress.py                            # every games-*.out
    python scripts/progress.py --configs config/runs/main # + each game's full total (queued tasks too)
    python scripts/progress.py 51880001 --tasks           # one array job, one line per task
    python scripts/progress.py --errors                   # what failed, grouped by error

Per task: the episodes it was given (its `... episode(s) to play in shard` line), how many are
done / failed, and how long ago it last logged (a long silence = stuck or dead). A resubmitted task
counts only what it was given this time; finished episodes from earlier submissions are in --configs.
"""
from __future__ import annotations

import argparse
import functools
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

TO_PLAY = re.compile(r"\] \d+ run\(s\), (\d+) episode\(s\) to play in shard \d+/\d+(?: of (\S+))?")
DONE = re.compile(r"^\[[\d:]+\] (\S+) episode \d+ done after", re.M)
FAILED = re.compile(r"^\[[\d:]+\] ERROR (\S+) episode", re.M)
# a failed episode's block: "[t] ERROR <run> episode <n> (<stage>):" + traceback, up to a blank line
ERROR_BLOCK = re.compile(r"^\[[\d:]+\] ERROR (\S+) episode (\d+) \(([^)]*)\):\n(.*?)(?:\n\s*\n|\Z)", re.M | re.S)
# an engine that died takes the whole task down without per-episode ERROR lines
FATAL = re.compile(r"(DeliveryReadTimeout|EngineDeadError|ApertureBackpressureError|MiaSizingError|CUDA out of memory|"
                   r"No space left on device|DUE TO TIME LIMIT|Engine core initialization failed)")


def errors(paths, sweep: str = "main") -> None:
    """Failed episodes grouped by their error (the traceback's last line), with where they come from."""
    groups = defaultdict(list)  # error -> [(task, run, episode, stage, traceback)]
    fatal = defaultdict(list)
    rerun = defaultdict(list)  # (config dir, number of shards) -> shard indices of tasks that died
    job_group = {}  # array job -> game group, from any of its tasks' run ids
    for path in paths:
        runs = DONE.findall(path.read_text(errors="replace"))
        if runs:
            job_group.setdefault(path.stem.removeprefix("games-").split("_")[0], runs[0].split("__")[0])
    for path in paths:
        text = path.read_text(errors="replace")
        task = path.stem.removeprefix("games-")
        for run, ep, stage, tb in ERROR_BLOCK.findall(text):
            last = [line for line in tb.strip().splitlines() if line.strip()][-1].strip() if tb.strip() else "?"
            groups[re.sub(r"0x[0-9a-f]+|\d{3,}", "N", last)[:200]].append((task, run, ep, stage, tb))
        for m in sorted(set(FATAL.findall(text))):
            fatal[m].append(task)
        if FATAL.search(text):
            shard = re.search(r"to play in shard (\d+)/(\d+)(?: of (\S+))?", text)
            if shard:
                job, idx = task.split("_")[0], shard.group(1)
                runs = DONE.findall(text) + FAILED.findall(text)  # run ids start with their game group
                group = runs[0].split("__")[0] if runs else job_group.get(job) or slurm_group(job)
                where = shard.group(3) or (f"config/runs/{sweep}/{group}" if group else f"<config dir of job {job}>")
                rerun[(where, shard.group(2), job)].append(idx)
    if fatal:
        print("Tasks that died (engine / node problems; their unfinished episodes rerun on resubmit):")
        for kind, tasks in sorted(fatal.items(), key=lambda kv: -len(kv[1])):
            print(f"  {len(tasks):>4} task(s)  {kind}:  {', '.join(tasks[:6])}{' ...' if len(tasks) > 6 else ''}")
        print("\nRelaunch them (finished episodes are skipped; only after the dead task has left the queue):")
        for (where, n, _), idx in sorted(rerun.items()):
            ids = ",".join(sorted(set(idx), key=int))
            print(f"  sbatch --array={ids} --export=ALL,NUM_SHARDS={n} scripts/slurm/run_games.sbatch {where}")
        print()
    if not groups:
        print("No failed episodes." if not fatal else "No per-episode failures.")
        return
    print(f"{sum(len(v) for v in groups.values())} failed episode(s), {len(groups)} distinct error(s):\n")
    for err, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        games = defaultdict(int)
        for _, run, *_ in items:
            games[run.split("__")[0]] += 1
        task, run, ep, stage, tb = items[0]
        print(f"[{len(items)}x] {err}")
        print(f"   games: {', '.join(f'{g} ({n})' for g, n in sorted(games.items(), key=lambda kv: -kv[1]))}")
        print(f"   e.g. {run} episode {ep} ({stage}), task {task}; full traceback:")
        print("   " + "\n   ".join(tb.strip().splitlines()[-8:]) + "\n")


EPISODES = re.compile(r"^episodes: (\d+)$", re.M)


def task_progress(path: Path) -> dict:
    text = path.read_text(errors="replace")
    to_play = TO_PLAY.search(text)
    done, failed = DONE.findall(text), FAILED.findall(text)
    if to_play and to_play.group(2):
        group = Path(to_play.group(2)).name
    elif done or failed:  # logs from before the group was printed: the group is the run id's prefix
        group = (done or failed)[0].split("__")[0]
    else:  # nothing finished yet: ask SLURM which config dir the array was submitted with
        group = slurm_group(path.stem.removeprefix("games-").split("_")[0]) or "(starting)"
    return {"task": path.stem.removeprefix("games-"), "group": group,
            "total": int(to_play.group(1)) if to_play else None, "done": len(done), "failed": len(failed),
            "idle_min": (time.time() - path.stat().st_mtime) / 60}


@functools.lru_cache(maxsize=None)
def slurm_group(job_id: str) -> str:
    """The game group of a running array job: the config dir its run_games.sbatch was given."""
    try:
        out = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    command = re.search(r"Command=(.*)", out)
    return Path(command.group(1).split()[-1]).name if command and len(command.group(1).split()) > 1 else ""


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
    p.add_argument("--errors", action="store_true", help="group the failures by error instead")
    p.add_argument("--sweep", default="main", help="sweep name, for the relaunch commands' config dirs")
    args = p.parse_args(argv)
    if args.errors:
        errors(sorted(Path(args.logs).glob(f"games-{args.job}*.out")), args.sweep)
        return 0

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
