"""One CSV row per (run, episode, agent): condition, steering, role/team, reward, monitor means.

    python scripts/summarize_runs.py logs/main [fallback/main] --out results/main_agents.csv [--layer 16]

Several log dirs (e.g. scratch and the MA_FALLBACK_LOG_DIR copy of a sweep) are merged per run; an episode
logged in both counts once, from the first dir given.

Monitor columns `mon_<trait>` are the agent's mean projection over its turns at one layer
(--layer; default: the middle monitored layer). Steering info comes from each run's run.json
and episode summary, never from the run_id.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def agent_rows(run_dirs, layer=None):
    """Rows of one run, logged in `run_dirs` (a dir or a list of dirs of the same run_id)."""
    run_dirs = [run_dirs] if isinstance(run_dirs, Path) else list(run_dirs)
    run = json.loads((run_dirs[0] / "run.json").read_text(encoding="utf-8"))
    steering = run.get("steering") or {}
    steering = steering if isinstance(steering, dict) else {"groups": len(steering)}
    episodes = {}  # episode -> its dir, the first dir that has it
    for d in run_dirs:
        for summary_path in sorted(d.glob("episode_*.summary.json")):
            episodes.setdefault(int(summary_path.name.split("_")[1].split(".")[0]), d)
    for ep, run_dir in sorted(episodes.items()):
        summary_path = run_dir / f"episode_{ep}.summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        turns, fallbacks, mon = defaultdict(int), defaultdict(int), defaultdict(lambda: defaultdict(list))
        chars = defaultdict(int)
        at = None
        jsonl = run_dir / f"episode_{ep}.jsonl"
        for line in jsonl.read_text(encoding="utf-8").splitlines() if jsonl.exists() else []:
            rec = json.loads(line)
            aid = rec["agent_id"]
            turns[aid] += 1
            fallbacks[aid] += not rec.get("parse_ok", True)
            chars[aid] += len(rec.get("completion") or "")
            if rec.get("monitor"):
                at = str(layer) if layer is not None else sorted(rec["monitor"], key=int)[len(rec["monitor"]) // 2]
                for trait, v in rec["monitor"].get(at, {}).items():
                    mon[aid][trait].append(v)
        agents = sorted(set(summary.get("final_rewards", {})) | set(turns))
        for aid in agents:
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
                "turns": turns[aid], "parse_fallbacks": fallbacks[aid],
                "answer_chars": chars[aid] / turns[aid] if turns[aid] else None,  # mean answer length
                "monitor_layer": at, "steer_layer": (run.get("steering_resolved") or [{}])[0].get("layer"),
                "episode_turns": summary.get("turns"), "truncated": summary.get("truncated"),
                **{f"mon_{t}": statistics.mean(v) for t, v in sorted(mon[aid].items())},
            }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="log dir(s) of one sweep; the first wins for an episode logged twice")
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int)
    args = p.parse_args(argv)
    runs = {}  # run dir relative to its log dir (<group>/<run_id>) -> [dirs that have it]
    for logs in map(Path, args.logs):
        if logs.is_dir():
            for run_json in sorted(logs.rglob("run.json")):
                runs.setdefault(run_json.parent.relative_to(logs), []).append(run_json.parent)
    rows = [r for _, dirs in sorted(runs.items()) for r in agent_rows(dirs, args.layer)]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} agent-episode rows -> {args.out}")


if __name__ == "__main__":
    main()
