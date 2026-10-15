"""One CSV row per (run, episode, agent): condition, steering, role/team, reward, monitor means.

    python scripts/summarize_runs.py logs/main [fallback/main] --out results/main_agents.csv [--layer 16]

Everything comes from the small episode summaries: rewards, roles, teams, who was steered how, and
`agent_stats` (per agent its turns, parse fallbacks, answer length and monitor means). Episodes logged
before summaries had agent_stats are read from their JSONL instead (every prompt of every turn: slow), and
those stats are cached next to --out (<out>.cache.pkl). Files are read in parallel (--jobs).
An --out ending in .gz is gzip-compressed (keeps a large sweep's CSV under GitHub's 100 MB file limit).

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
from pathlib import Path


def episode_stats(jsonl) -> dict:
    """agent_stats as the logger writes them into the summary, recomputed from an episode's JSONL (for
    episodes logged before): {agent: {turns, fallbacks, chars, monitor: {layer: {trait: mean}}}}."""
    stats = {}
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            st = stats.setdefault(rec["agent_id"], {"turns": 0, "fallbacks": 0, "chars": 0, "monitor": {}})
            st["turns"] += 1
            st["fallbacks"] += not rec.get("parse_ok", True)
            st["chars"] += len(rec.get("completion") or "")
            for layer, scores in (rec.get("monitor") or {}).items():
                for trait, v in scores.items():
                    st["monitor"].setdefault(str(layer), {}).setdefault(trait, []).append(v)
    for st in stats.values():
        st["monitor"] = {layer: {t: statistics.mean(v) for t, v in traits.items()}
                         for layer, traits in st["monitor"].items()}
    return stats


def run_episodes(run_dirs) -> dict:
    """episode -> its dir, the first of `run_dirs` that has its summary."""
    episodes = {}
    for d in run_dirs:
        for summary_path in sorted(d.glob("episode_*.summary.json")):
            episodes.setdefault(int(summary_path.name.split("_")[1].split(".")[0]), d)
    return episodes


def read_episode(job):
    """(summary path, cached (JSONL size, agent_stats) or None) -> (summary path, summary with agent_stats,
    (size, agent_stats) if the JSONL had to be read else None)."""
    spath, cached = job
    summary = json.loads(Path(spath).read_text(encoding="utf-8"))
    if "agent_stats" in summary:
        return spath, summary, None
    jsonl = Path(spath[:-len(".summary.json")] + ".jsonl")
    try:
        size = jsonl.stat().st_size
    except OSError:
        return spath, {**summary, "agent_stats": {}}, None
    if cached and cached[0] == size:
        return spath, {**summary, "agent_stats": cached[1]}, None
    stats = episode_stats(jsonl)
    return spath, {**summary, "agent_stats": stats}, (size, stats)


def agent_rows(run_dirs, layer=None, summaries=None):
    """Rows of one run, logged in `run_dirs` (a dir or a list of dirs of the same run_id). `summaries`:
    summary path (str) -> summary with agent_stats; read here for those missing."""
    run_dirs = [run_dirs] if isinstance(run_dirs, Path) else list(run_dirs)
    run = json.loads((run_dirs[0] / "run.json").read_text(encoding="utf-8"))
    steering = run.get("steering") or {}
    steering = steering if isinstance(steering, dict) else {"groups": len(steering)}
    summaries = {} if summaries is None else summaries
    for ep, run_dir in sorted(run_episodes(run_dirs).items()):
        spath = str(run_dir / f"episode_{ep}.summary.json")
        if spath not in summaries:
            summaries[spath] = read_episode((spath, None))[1]
        summary = summaries[spath]
        stats = summary["agent_stats"]
        layers = sorted({lay for a in stats.values() for lay in a["monitor"]}, key=int)
        at = str(layer) if layer is not None else (layers[len(layers) // 2] if layers else None)
        none = {"turns": 0, "fallbacks": 0, "chars": 0, "monitor": {}}
        for aid in sorted(set(summary.get("final_rewards", {})) | set(stats)):
            a = stats.get(aid, none)
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
                **{f"mon_{t}": v for t, v in sorted(a["monitor"].get(at, {}).items())},
            }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="log dir(s) of one sweep; the first wins for an episode logged twice")
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int)
    p.add_argument("--jobs", type=int, default=8, help="processes reading episode files in parallel")
    args = p.parse_args(argv)
    runs = {}  # run dir relative to its log dir (<group>/<run_id>) -> [dirs that have it]
    for logs in map(Path, args.logs):
        if logs.is_dir():
            for run_json in sorted(logs.rglob("run.json")):
                runs.setdefault(run_json.parent.relative_to(logs), []).append(run_json.parent)
    spaths = [str(d / f"episode_{ep}.summary.json")
              for _, dirs in sorted(runs.items()) for ep, d in run_episodes(dirs).items()]
    print(f"{len(runs)} runs, {len(spaths)} finished episodes", flush=True)
    # agent_stats of episodes whose summary lacks them, from their JSONL: summary path -> (JSONL size, stats)
    cache_path = Path(args.out + ".cache.pkl")
    cache = {}
    if cache_path.exists():
        with gzip.open(cache_path, "rb") as f:
            cache = pickle.load(f)

    def save():
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_path, "wb") as f:
            pickle.dump(cache, f)

    summaries, n_read = {}, 0
    with multiprocessing.Pool(args.jobs) as pool:
        work = pool.imap_unordered(read_episode, [(sp, cache.get(sp)) for sp in spaths], chunksize=16)
        for i, (spath, summary, read) in enumerate(work, 1):
            summaries[spath] = summary
            if read:
                cache[spath] = read
                n_read += 1
                if n_read % 1000 == 0:
                    save()
            if i % 2000 == 0 or i == len(spaths):
                print(f"  {i}/{len(spaths)} episodes ({n_read} older ones read from their JSONL)", flush=True)
    if n_read:
        save()
    rows = [r for _, dirs in sorted(runs.items()) for r in agent_rows(dirs, args.layer, summaries)]
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
