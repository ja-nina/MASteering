"""The games' own scores (tokens, chips, points) for TextArena episodes logged before summaries had `game_scores`.

    python scripts/backfill_scores.py <logs>/main_v2 [fallback/main_v2] [--jobs 8] [--games public_goods_4p ...]

No model runs. Each finished episode is replayed: the game is rebuilt from the run's run.json with the episode's seed
(the same build_game and adapter as the run), and the moves the agents made, as logged in the JSONL (parsed_action:
exactly what went to the game, after the ACTION marker), are submitted again in their logged order. The game's score
is then read as the run now does at its end (testbed/envs/textarena/scores.py).

A replay is kept only when it reproduces the logged episode: the same player asked at every turn, the same number of
turns, and the same final win/lose rewards as the summary. Games that draw random numbers DURING play (poker's later
hands, Catan's dice) used Python's global random, which other episodes of the same process also drew from: those
replays usually diverge and are reported, not kept.

Written next to the summary: episode_N.scores.json {"game_scores": {agent: score}, "score": <what it is>,
"source": "replay"}. scripts/summarize_runs.py takes the score from the summary's game_scores, else from this file.
Episodes that have either are skipped, so it can be rerun.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def side_path(summary_path: Path) -> Path:
    return summary_path.with_name(summary_path.name[:-len(".summary.json")] + ".scores.json")


def replay(run: dict, episode: int, jsonl: Path, summary: dict) -> tuple:
    """(scores or None, status)."""
    from testbed.config import RunConfig
    from testbed.envs.textarena.scores import score_name
    from testbed.registry import build_game

    cfg = RunConfig.from_dict(run)
    env, _, _ = build_game(family=cfg.game_family, game_id=cfg.game_id, num_players=cfg.players,
                           env_kwargs={"seed": cfg.episode_seed(episode), **cfg.env_kwargs})
    env.reset()
    records = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    turns, rewards, done = 0, {}, False
    for _, group in groupby(records, key=lambda r: r["turn"]):
        group = list(group)
        asked = sorted(a for a, _ in env.pending())
        actions = {r["agent_id"]: r["parsed_action"] for r in group}
        if sorted(actions) != asked:
            return None, f"diverged at turn {turns}: game asks {asked}, log has {sorted(actions)}"
        step = env.submit(actions)
        turns += 1
        if step.done:
            rewards, done = step.rewards, True
            break
    if turns != summary.get("turns"):
        return None, f"turns {turns} != logged {summary.get('turns')}"
    if done and {k: float(v) for k, v in rewards.items()} != \
            {k: float(v) for k, v in (summary.get("final_rewards") or {}).items()}:
        return None, "final rewards differ from the log"
    scores = env.game_scores()
    if scores is None:
        return None, f"no score readable ({score_name(cfg.game_id) or 'game has none'})"
    return {"game_scores": scores, "score": score_name(cfg.game_id), "source": "replay"}, "ok"


def work(job) -> tuple:
    run, ep, spath = job
    spath = Path(spath)
    try:
        summary = json.loads(spath.read_text(encoding="utf-8"))
        jsonl = spath.with_name(f"episode_{ep}.jsonl")
        if not jsonl.exists():
            return run["run_id"], "no jsonl"
        out, status = replay(run, ep, jsonl, summary)
        if out is not None:
            side_path(spath).write_text(json.dumps(out, indent=1), encoding="utf-8")
        return run["run_id"], status
    except Exception as e:  # noqa: BLE001 - one bad episode must not stop the backfill
        return run["run_id"], f"error: {type(e).__name__}: {e}"[:200]


def main(argv=None) -> None:
    from testbed.envs.textarena.scores import SCORES, base_id

    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--games", nargs="*", help="game groups (dir names) to do; default every game with a score")
    a = ap.parse_args(argv)
    jobs, have = [], 0
    for d in a.logs:
        for run_json in sorted(Path(d).glob("*/*/run.json")):
            if a.games and run_json.parent.parent.name not in a.games:
                continue
            run = json.loads(run_json.read_text(encoding="utf-8"))
            game = run.get("game") or {}
            if game.get("family") != "textarena" or base_id(game.get("id", "")) not in SCORES:
                continue
            for spath in sorted(run_json.parent.glob("episode_*.summary.json")):
                ep = int(spath.name.split("_")[1].split(".")[0])
                if side_path(spath).exists() or json.loads(spath.read_text(encoding="utf-8")).get("game_scores"):
                    have += 1
                    continue
                jobs.append((run, ep, str(spath)))
    print(f"{len(jobs)} episodes to replay ({have} already have scores)", flush=True)
    status = Counter()
    by_game = Counter()
    pool = multiprocessing.Pool(a.jobs) if a.jobs > 1 else None  # --jobs 1: in this process (tests, debugging)
    results = pool.imap_unordered(work, jobs, chunksize=4) if pool else map(work, jobs)
    try:
        for i, (run_id, st) in enumerate(results, 1):
            kind = "ok" if st == "ok" else st.split(":")[0].split(" at ")[0]
            status[kind] += 1
            by_game[(run_id.split("__")[0], kind)] += 1
            if st != "ok" and status[kind] <= 3:
                print(f"  {run_id}: {st}", flush=True)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}  {dict(status)}", flush=True)
    finally:
        if pool:
            pool.close()
    print(f"done: {dict(status)}")
    for (g, kind), n in sorted(by_game.items()):
        print(f"  {g:28s} {kind:30s} {n}")


if __name__ == "__main__":
    main()
