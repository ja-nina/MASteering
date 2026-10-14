"""One CSV row per (run, episode, agent): condition, steering, role/team, reward, monitor means.

    python scripts/summarize_runs.py logs/main --out results/main_agents.csv [--layer 16]

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


def agent_rows(run_dir: Path, layer=None):
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    steering = run.get("steering") or {}
    steering = steering if isinstance(steering, dict) else {"groups": len(steering)}
    for summary_path in sorted(run_dir.glob("episode_*.summary.json")):
        ep = int(summary_path.name.split("_")[1].split(".")[0])
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
    p.add_argument("logs")
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int)
    args = p.parse_args(argv)
    rows = [r for run_json in sorted(Path(args.logs).rglob("run.json"))
            for r in agent_rows(run_json.parent, args.layer)]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} agent-episode rows -> {args.out}")


if __name__ == "__main__":
    main()
