"""One CSV row per (run, episode, agent): condition, steering, role/team, reward, monitor means.

    python scripts/summarize_runs.py logs/main [fallback/main] --out results/main_agents.csv [--layer 16]

Everything comes from the small episode summaries: rewards, roles, teams, who was steered how, and
`agent_stats` (per agent its turns, parse fallbacks, answer length and monitor means). Episodes logged
before summaries had agent_stats get them once from their JSONL (every prompt of every turn: slow), written
back into the summary, so the next run reads only summaries. Episodes are read in parallel (--jobs) and rows
streamed to disk, so memory stays flat however large the sweep.
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
import os
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


def read_summary(spath: Path) -> tuple:
    """(summary with agent_stats, whether they came from the JSONL). Those are written back into the
    summary file (atomically; skipped if it is not writable), so the JSONL is read only once."""
    summary = json.loads(spath.read_text(encoding="utf-8"))
    if "agent_stats" in summary:
        return summary, False
    jsonl = spath.with_name(spath.name[:-len(".summary.json")] + ".jsonl")
    summary["agent_stats"] = episode_stats(jsonl) if jsonl.exists() else {}
    tmp = spath.with_name(f"{spath.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(spath)
    except OSError:
        tmp.unlink(missing_ok=True)
    return summary, True


def episode_rows(job) -> tuple:
    """(run.json, episode, summary path, layer) -> (that episode's rows, whether its JSONL was read)."""
    run, ep, spath, layer = job
    summary, from_jsonl = read_summary(Path(spath))
    steering = run.get("steering") or {}
    steering = steering if isinstance(steering, dict) else {"groups": len(steering)}
    stats = summary["agent_stats"]
    layers = sorted({lay for a in stats.values() for lay in a["monitor"]}, key=int)
    at = str(layer) if layer is not None else (layers[len(layers) // 2] if layers else None)
    none = {"turns": 0, "fallbacks": 0, "chars": 0, "monitor": {}}
    rows = []
    for aid in sorted(set(summary.get("final_rewards", {})) | set(stats)):
        a = stats.get(aid, none)
        steer = summary.get("steered", {}).get(aid)
        rows.append({
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
        })
    return rows, from_jsonl


def run_jobs(run_dirs, layer=None) -> list:
    """One episode_rows job per finished episode of a run logged in `run_dirs` (a dir or a list of dirs)."""
    run_dirs = [run_dirs] if isinstance(run_dirs, Path) else list(run_dirs)
    run = json.loads((run_dirs[0] / "run.json").read_text(encoding="utf-8"))
    return [(run, ep, str(d / f"episode_{ep}.summary.json"), layer) for ep, d in sorted(run_episodes(run_dirs).items())]


def agent_rows(run_dirs, layer=None):
    """Rows of one run, logged in `run_dirs` (a dir or a list of dirs of the same run_id)."""
    for job in run_jobs(run_dirs, layer):
        yield from episode_rows(job)[0]


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="log dir(s) of one sweep; the first wins for an episode logged twice")
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int)
    p.add_argument("--jobs", type=int, default=8, help="processes reading episodes in parallel")
    args = p.parse_args(argv)
    runs = {}  # run dir relative to its log dir (<group>/<run_id>) -> [dirs that have it]
    for logs in map(Path, args.logs):
        if logs.is_dir():
            for run_json in sorted(logs.rglob("run.json")):
                runs.setdefault(run_json.parent.relative_to(logs), []).append(run_json.parent)
    jobs = [job for _, dirs in sorted(runs.items()) for job in run_jobs(dirs, args.layer)]
    print(f"{len(runs)} runs, {len(jobs)} finished episodes", flush=True)
    # rows go to a temp file as they come (the columns are known only at the end), then into the CSV
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out + ".rows.tmp")
    fields, n_rows, n_jsonl = {}, 0, 0
    with gzip.open(tmp, "wt", encoding="utf-8") as t, multiprocessing.Pool(args.jobs) as pool:
        for i, (rows, from_jsonl) in enumerate(pool.imap(episode_rows, jobs, chunksize=16), 1):
            n_jsonl += from_jsonl
            for r in rows:
                fields.update(dict.fromkeys(r))
                t.write(json.dumps(r, default=str) + "\n")
            n_rows += len(rows)
            if i % 2000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} episodes ({n_jsonl} older ones read from their JSONL, "
                      f"now in their summary)", flush=True)
    opener = gzip.open if args.out.endswith(".gz") else open
    with gzip.open(tmp, "rt", encoding="utf-8") as t, opener(args.out, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for line in t:
            writer.writerow(json.loads(line))
    tmp.unlink()
    print(f"{n_rows} agent-episode rows -> {args.out}")


if __name__ == "__main__":
    main()
