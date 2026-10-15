"""One CSV row per (run, episode, agent): condition, steering, role/team, reward, monitor means.

    python scripts/summarize_runs.py logs/main [fallback/main] --out results/main_agents.csv [--layer 16]

Episode logs are read in parallel (--jobs) and cached next to --out (<out>.cache.pkl): a rerun reads only
new episodes. An --out ending in .gz is gzip-compressed (keeps a large sweep's CSV under GitHub's 100 MB file limit).

Several log dirs (e.g. scratch and the MA_FALLBACK_LOG_DIR copy of a sweep) are merged per run; an episode
logged in both counts once, from the first dir given.

Monitor columns `mon_<trait>` are the agent's mean projection over its turns at one layer
(--layer; default: the middle monitored layer). Steering info comes from each run's run.json
and episode summary, never from the run_id.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing
import pickle
import statistics
from collections import defaultdict
from pathlib import Path


def episode_stats(jsonl, layer=None) -> dict:
    """Per agent of one episode's JSONL: turns, parse fallbacks, answer chars and mean monitor projection
    per trait at one layer; "_at" = that layer. Reading the JSONL (every prompt of every turn) is the slow
    part, so main() runs this in parallel and caches it."""
    turns, fallbacks, mon = defaultdict(int), defaultdict(int), defaultdict(lambda: defaultdict(list))
    chars = defaultdict(int)
    at = None
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            aid = rec["agent_id"]
            turns[aid] += 1
            fallbacks[aid] += not rec.get("parse_ok", True)
            chars[aid] += len(rec.get("completion") or "")
            if rec.get("monitor"):
                at = str(layer) if layer is not None else sorted(rec["monitor"], key=int)[len(rec["monitor"]) // 2]
                for trait, v in rec["monitor"].get(at, {}).items():
                    mon[aid][trait].append(v)
    return {"_at": at, **{aid: {"turns": turns[aid], "fallbacks": fallbacks[aid], "chars": chars[aid],
                                "mon": {t: statistics.mean(v) for t, v in sorted(mon[aid].items())}}
                          for aid in turns}}


def run_episodes(run_dirs) -> dict:
    """episode -> its dir, the first of `run_dirs` that has its summary."""
    episodes = {}
    for d in run_dirs:
        for summary_path in sorted(d.glob("episode_*.summary.json")):
            episodes.setdefault(int(summary_path.name.split("_")[1].split(".")[0]), d)
    return episodes


def agent_rows(run_dirs, layer=None, stats=None):
    """Rows of one run, logged in `run_dirs` (a dir or a list of dirs of the same run_id). `stats`: the
    episode_stats of each JSONL path (str), computed here for those missing."""
    run_dirs = [run_dirs] if isinstance(run_dirs, Path) else list(run_dirs)
    run = json.loads((run_dirs[0] / "run.json").read_text(encoding="utf-8"))
    steering = run.get("steering") or {}
    steering = steering if isinstance(steering, dict) else {"groups": len(steering)}
    stats = {} if stats is None else stats
    for ep, run_dir in sorted(run_episodes(run_dirs).items()):
        summary = json.loads((run_dir / f"episode_{ep}.summary.json").read_text(encoding="utf-8"))
        jsonl = run_dir / f"episode_{ep}.jsonl"
        if str(jsonl) not in stats:
            stats[str(jsonl)] = episode_stats(jsonl, layer) if jsonl.exists() else {"_at": None}
        st = stats[str(jsonl)]
        at = st["_at"]
        none = {"turns": 0, "fallbacks": 0, "chars": 0, "mon": {}}
        turns = {aid: a["turns"] for aid, a in st.items() if aid != "_at"}
        agents = sorted(set(summary.get("final_rewards", {})) | set(turns))
        for aid in agents:
            a = st.get(aid, none)
            steer = summary.get("steered", {}).get(aid)
            yield {
                "run_id": run["run_id"], "group": run["run_id"].split("__")[0],  # <game>_<n>p, as in the sweep
                "condition": run["run_id"].split("__")[1] if "__" in run["run_id"] else "none",
                "game": run["game"]["id"], "players": run["game"]["players"],
                "observation": run.get("observation", "text"), "episode": ep, "agent": aid,
                "condition_select": steering.get("select", "all" if steering else None),
                "condition_team": steering.get("team"), "condition_trait": steering.get("trait"),
                "condition_coefficient": steering.get("coefficient"),
                "condition_via": steering.get("via", "vector") if steering else None,
                "condition_random": bool(steering.get("random_control", False)),
                "steered": steer is not None, "steer_label": steer["label"] if steer else None,
                "role": summary.get("player_roles", {}).get(aid), "team": summary.get("teams", {}).get(aid),
                "reward": summary.get("final_rewards", {}).get(aid),
                "turns": a["turns"], "parse_fallbacks": a["fallbacks"],
                "answer_chars": a["chars"] / a["turns"] if a["turns"] else None,  # mean answer length
                "monitor_layer": at, "steer_layer": (run.get("steering_resolved") or [{}])[0].get("layer"),
                "episode_turns": summary.get("turns"), "truncated": summary.get("truncated"),
                **{f"mon_{t}": v for t, v in a["mon"].items()},
            }


def _read(job):
    path, size, layer = job
    return path, size, episode_stats(path, layer)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="log dir(s) of one sweep; the first wins for an episode logged twice")
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int)
    p.add_argument("--jobs", type=int, default=8, help="processes reading episode JSONLs in parallel")
    args = p.parse_args(argv)
    runs = {}  # run dir relative to its log dir (<group>/<run_id>) -> [dirs that have it]
    for logs in map(Path, args.logs):
        if logs.is_dir():
            for run_json in sorted(logs.rglob("run.json")):
                runs.setdefault(run_json.parent.relative_to(logs), []).append(run_json.parent)
    print(f"{len(runs)} runs found", flush=True)
    # episode_stats of every JSONL already read, keyed by path; a JSONL whose size changed is read again
    cache_path = Path(args.out + ".cache.pkl")
    cache = {}
    if cache_path.exists():
        with gzip.open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("_layer") != args.layer:
            cache = {}
    todo = []
    for _, dirs in sorted(runs.items()):
        for ep, d in run_episodes(dirs).items():
            jsonl = d / f"episode_{ep}.jsonl"
            try:
                size = jsonl.stat().st_size
            except OSError:
                continue
            if cache.get(str(jsonl), (None,))[0] != size:
                todo.append((str(jsonl), size))
    print(f"{len(todo)} episode logs to read ({len(cache) - ('_layer' in cache)} cached in {cache_path})", flush=True)

    def save():
        cache["_layer"] = args.layer
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    with multiprocessing.Pool(args.jobs) as pool:
        work = pool.imap_unordered(_read, [(path, size, args.layer) for path, size in todo], chunksize=4)
        for i, (path, size, st) in enumerate(work, 1):
            cache[path] = (size, st)
            if i % 500 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} episode logs read ({path.split('/')[-3] if '/' in path else ''})", flush=True)
            if i % 5000 == 0:
                save()
    save()
    stats = {k: v[1] for k, v in cache.items() if k != "_layer"}
    rows = [r for _, dirs in sorted(runs.items()) for r in agent_rows(dirs, args.layer, stats)]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.out.endswith(".gz") else open
    with opener(args.out, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} agent-episode rows -> {args.out}")


if __name__ == "__main__":
    main()
