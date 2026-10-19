"""One CSV row per (run, episode, agent): condition, steering, role/team, reward, monitor means.

    python scripts/summarize_runs.py logs/main [fallback/main] --out results/main_agents.csv [--layer 16]

Everything comes from the small episode summaries: rewards, roles, teams, who was steered how, and
`agent_stats` (testbed/logging_/agent_stats.py). `reward` is the episode's score: the summary's `total_rewards`
(every step's rewards summed); for episodes logged before summaries had it, the episode_N.rewards.json that
scripts/backfill_rewards.py writes for the games that pay every step, else `final_rewards` (the last step's
rewards: the score in games that pay once, at the end). `reward_from` says which: total | backfill | last_step. Episodes logged before summaries had them (or before they
had phases and actions) get them once from their JSONL (every prompt of every turn: slow), written back into
the summary, so the next run reads only summaries. Episodes are read in parallel (--jobs) and rows
streamed to disk, so memory stays flat however large the sweep.
An --out ending in .gz is gzip-compressed and, past MAX_PART_BYTES, continued in parts <name>-2.csv.gz,
<name>-3.csv.gz, ... next to it (each a full CSV with header), so every file stays under GitHub's 100 MB limit;
`csv_parts(out)` lists them (scripts/report.py reads them all).

Several log dirs (e.g. scratch and the MA_FALLBACK_LOG_DIR copy of a sweep) are merged per run; an episode
logged in both counts once, from the first dir given.

Monitor columns `mon_<trait>` are the agent's mean projection over its turns at one layer
(--layer; default: the middle monitored layer); `mon_early_/mon_mid_/mon_late_<trait>` the same over the first,
middle and last third of its turns (steered traits only, middle layer). Actions: `act_n` parsed choices,
`act_numeric` their numeric mean (bids, guesses, amounts), `act_labels` JSON {action label: count}. Steering info comes from each run's run.json
and episode summary, never from the run_id.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.backfill_rewards import PER_STEP_GAMES, side_path  # noqa: E402
from testbed.logging_.agent_stats import stats_from_jsonl  # noqa: E402

MAX_PART_BYTES = 90_000_000  # compressed size of one part of a .gz output


def part_path(out, i: int) -> Path:
    """Part i (1-based) of the output `out`: part 1 is `out` itself, then <name>-2.csv.gz, ..."""
    out = Path(out)
    return out if i == 1 else out.with_name(f"{out.name[:-len('.csv.gz')]}-{i}.csv.gz")


def csv_parts(out) -> list:
    """The files of an output written by main(): `out`, then its parts as long as they exist."""
    parts = [Path(out)]
    while str(out).endswith(".csv.gz") and part_path(out, len(parts) + 1).exists():
        parts.append(part_path(out, len(parts) + 1))
    return parts


def write_csv(out: str, fields: list, rows) -> int:
    """Rows into the CSV `out`; a .csv.gz is split into parts of at most ~MAX_PART_BYTES. Number of files."""
    if not out.endswith(".csv.gz"):
        with (gzip.open if out.endswith(".gz") else open)(out, "wt", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return 1
    for stale in csv_parts(out)[1:]:  # parts of an earlier, larger output
        stale.unlink()

    def start(i):
        raw = open(part_path(out, i), "wb")
        f = io.TextIOWrapper(gzip.GzipFile(fileobj=raw, mode="wb"), encoding="utf-8", newline="")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        return raw, f, writer

    n = 1
    raw, f, writer = start(n)
    for i, row in enumerate(rows, 1):
        if i % 5000 == 0 and raw.tell() > MAX_PART_BYTES:
            f.close()
            raw.close()
            n += 1
            raw, f, writer = start(n)
        writer.writerow(row)
    f.close()
    raw.close()
    return n


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
    stats = summary.get("agent_stats")
    if stats is not None and all("actions" in a for a in stats.values()):
        return summary, False
    jsonl = spath.with_name(spath.name[:-len(".summary.json")] + ".jsonl")
    summary["agent_stats"] = stats_from_jsonl(jsonl) if jsonl.exists() else {}
    tmp = spath.with_name(f"{spath.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(spath)
    except OSError:
        tmp.unlink(missing_ok=True)
    return summary, True


def episode_rewards(summary: dict, spath: Path) -> tuple:
    """({agent: episode score}, where it came from: total | backfill | last_step)."""
    if "total_rewards" in summary:
        return summary["total_rewards"], "total"
    side = side_path(spath)
    if side.exists():
        return json.loads(side.read_text(encoding="utf-8"))["total_rewards"], "backfill"
    return summary.get("final_rewards", {}), "last_step"


def episode_tokens(stats: dict, spath: Path) -> dict:
    """{agent: (tokens read, tokens generated)}: from the summary's agent_stats, else from the episode_N.tokens.json
    that scripts/backfill_tokens.py writes for episodes logged before the counts; (None, None) when neither has it."""
    side = spath.with_name(spath.name[:-len(".summary.json")] + ".tokens.json")
    backfill = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    out = {}
    for aid, a in stats.items():
        if a.get("tokens_in") is not None:
            out[aid] = (a["tokens_in"], a.get("tokens_out"))
        else:
            b = backfill.get(aid) or {}
            out[aid] = (b.get("in"), b.get("out", a.get("tokens_out")))
    return out


def episode_rows(job) -> tuple:
    """(run.json, episode, summary path, layer, steered traits) -> (that episode's rows, whether its JSONL was read)."""
    run, ep, spath, layer, traits = job
    summary, from_jsonl = read_summary(Path(spath))
    steering = run.get("steering") or {}
    steering = steering if isinstance(steering, dict) else {"groups": len(steering)}
    stats = summary["agent_stats"]
    layers = sorted({lay for a in stats.values() for lay in a["monitor"]}, key=int)
    at = str(layer) if layer is not None else (layers[len(layers) // 2] if layers else None)
    none = {"turns": 0, "fallbacks": 0, "chars": 0, "monitor": {}, "phases": [None] * 3, "actions": {}}
    rewards, reward_from = episode_rewards(summary, Path(spath))
    tokens = episode_tokens(stats, Path(spath))
    rows = []
    for aid in sorted(set(rewards) | set(stats)):
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
            "reward": rewards.get(aid), "reward_from": reward_from,
            "turns": a["turns"], "parse_fallbacks": a["fallbacks"],
            "answer_chars": a["chars"] / a["turns"] if a["turns"] else None,  # mean answer length
            "tokens_in": tokens.get(aid, (None, None))[0], "tokens_out": tokens.get(aid, (None, None))[1],
            "monitor_layer": at, "steer_layer": (run.get("steering_resolved") or [{}])[0].get("layer"),
            "episode_turns": summary.get("turns"), "truncated": summary.get("truncated"),
            **{f"mon_{t}": v for t, v in sorted(a["monitor"].get(at, {}).items())},
            **({f"mon_{name}_{t}": ph[t] for name, ph in zip(("early", "mid", "late"), a.get("phases") or [])
                if ph for t in traits if t in ph} if a.get("phase_layer") == at else {}),
            "act_n": (a.get("actions") or {}).get("n"), "act_numeric": (a.get("actions") or {}).get("numeric"),
            "act_labels": json.dumps((a.get("actions") or {}).get("labels") or {}, separators=(",", ":")),
        })
    return rows, from_jsonl


def run_jobs(run_dirs, layer=None, traits=()) -> list:
    """One episode_rows job per finished episode of a run logged in `run_dirs` (a dir or a list of dirs)."""
    run_dirs = [run_dirs] if isinstance(run_dirs, Path) else list(run_dirs)
    run = json.loads((run_dirs[0] / "run.json").read_text(encoding="utf-8"))
    return [(run, ep, str(d / f"episode_{ep}.summary.json"), layer, tuple(traits))
            for ep, d in sorted(run_episodes(run_dirs).items())]


def steered_trait(run) -> str | None:
    steering = run.get("steering")
    return steering.get("trait") if isinstance(steering, dict) else None


def agent_rows(run_dirs, layer=None, traits=()):
    """Rows of one run, logged in `run_dirs` (a dir or a list of dirs of the same run_id)."""
    for job in run_jobs(run_dirs, layer, traits):
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
    traits = tuple(sorted({t for job in jobs if (t := steered_trait(job[0]))}))  # phase columns: these only
    jobs = [job[:4] + (traits,) for job in jobs]
    print(f"{len(runs)} runs, {len(jobs)} finished episodes", flush=True)
    # rows go to a temp file as they come (the columns are known only at the end), then into the CSV
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out + ".rows.tmp")
    fields, n_rows, n_jsonl, n_last = {}, 0, 0, 0
    with gzip.open(tmp, "wt", encoding="utf-8") as t, multiprocessing.Pool(args.jobs) as pool:
        for i, (rows, from_jsonl) in enumerate(pool.imap(episode_rows, jobs, chunksize=16), 1):
            n_jsonl += from_jsonl
            for r in rows:
                n_last += r["game"] in PER_STEP_GAMES and r["reward_from"] == "last_step"
                fields.update(dict.fromkeys(r))
                t.write(json.dumps(r, default=str) + "\n")
            n_rows += len(rows)
            if i % 2000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} episodes ({n_jsonl} older ones read from their JSONL, "
                      f"now in their summary)", flush=True)
    with gzip.open(tmp, "rt", encoding="utf-8") as t:
        n_files = write_csv(args.out, list(fields), (json.loads(line) for line in t))
    tmp.unlink()
    if n_last:
        print(f"WARNING: {n_last} rows of games that pay every step ({', '.join(sorted(PER_STEP_GAMES))}) "
              f"have only their last step's reward: run scripts/backfill_rewards.py on these log dirs first")
    print(f"{n_rows} agent-episode rows -> {args.out}" + (f" (+ {n_files - 1} more part(s))" if n_files > 1 else ""))


if __name__ == "__main__":
    main()
