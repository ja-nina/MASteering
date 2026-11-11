"""Debug helper (main_v2, 2026-11-11): why taboo team scores are ~always 0.

    python scripts/debug_taboo.py <logs>/main_v2 [fallback/main_v2] [--max 2000] [--examples 15]

TextArena's Taboo counts a guess only when the text inside the first [...] equals the word exactly
(`match.group(1).strip().lower() == word.lower()`), but its own instructions show the format as ['elephant'], with
quotes, which are never stripped: a model copying the example sends "'treadmill'" and can never score.

Each episode is replayed as in backfill_scores.py (seed + logged parsed_action, no model); before every guess the
word the team is guessing is read from the game. Each guess is put in one bucket:

  exact            counted by the game
  quotes only      would have counted without quote / punctuation characters around it
  word in answer   the word appears somewhere in the answer, not as the bracketed guess
  wrong            a real guess, not the word
  no brackets      no [...] in the answer (invalid move)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GUESS = re.compile(r"\[(.*?)\]")
STRIP = " \t\n'\"`.,!?*_"


def bucket(action: str, word: str) -> tuple:
    m = GUESS.search(action or "")
    w = word.lower()
    if not m:
        return ("word in answer" if w in (action or "").lower() else "no brackets"), None
    g = m.group(1).strip().lower()
    if g == w:
        return "exact", g
    if g.strip(STRIP) == w:
        return "quotes only", g
    if w in (action or "").lower():
        return "word in answer", g
    return "wrong", g


def episode(run: dict, ep: int, jsonl: Path):
    from testbed.config import RunConfig
    from testbed.envs.textarena.scores import _unwrap
    from testbed.registry import build_game

    cfg = RunConfig.from_dict(run)
    env, _, _ = build_game(family=cfg.game_family, game_id=cfg.game_id, num_players=cfg.players,
                           env_kwargs={"seed": cfg.episode_seed(ep), **cfg.env_kwargs})
    env.reset()
    records = [json.loads(line) for line in open(jsonl, encoding="utf-8") if line.strip()]
    out = []
    for _, group in groupby(records, key=lambda r: r["turn"]):
        group = list(group)
        inner = _unwrap(env._env)
        gs, roles = inner.state.game_state, inner.state.role_mapping
        for r in group:
            pid = int(r["agent_id"].split("_")[1])
            if roles.get(pid) == "Guesser":
                out.append((gs["word_to_guess"], r["parsed_action"], r.get("completion", "")))
        step = env.submit({r["agent_id"]: r["parsed_action"] for r in group})
        if step.done:
            break
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--max", type=int, default=2000, help="episodes to replay (random sample)")
    ap.add_argument("--examples", type=int, default=15)
    a = ap.parse_args(argv)
    eps = []
    for d in a.logs:
        for run_json in sorted(Path(d).glob("taboo_*/*/run.json")):
            run = json.loads(run_json.read_text(encoding="utf-8"))
            for j in run_json.parent.glob("episode_*.jsonl"):
                if j.with_name(j.name.replace(".jsonl", ".summary.json")).exists():
                    eps.append((run, int(j.name.split("_")[1].split(".")[0]), j))
    random.Random(0).shuffle(eps)
    eps = eps[:a.max]
    print(f"{len(eps)} taboo episodes replayed")
    total, unsteered, examples, words = Counter(), Counter(), {}, Counter()
    for run, ep, j in eps:
        try:
            guesses = episode(run, ep, j)
        except Exception as e:  # noqa: BLE001
            total["replay error"] += 1
            if total["replay error"] <= 3:
                print(f"  {run['run_id']} ep{ep}: {type(e).__name__}: {e}"[:200])
            continue
        for word, action, completion in guesses:
            b, g = bucket(action, word)
            total[b] += 1
            if run["run_id"].endswith("__none"):
                unsteered[b] += 1
            if b in ("quotes only", "word in answer"):
                words[word] += 1
            examples.setdefault(b, []).append((run["run_id"], word, g, action))
    for name, c in (("all runs", total), ("unsteered", unsteered)):
        n = sum(v for k, v in c.items() if k != "replay error")
        print(f"\n== guesses, {name} (n={n})")
        for k, v in c.most_common():
            print(f"  {k:16s} {v:7d}  {v / max(n, 1):6.1%}")
    rng = random.Random(1)
    for b in ("quotes only", "word in answer", "wrong", "no brackets", "exact"):
        ex = examples.get(b, [])
        if not ex:
            continue
        print(f"\n== examples: {b}")
        for run_id, word, g, action in rng.sample(ex, min(a.examples, len(ex))):
            print(f"  [{run_id}] word={word!r} guess={g!r}  answer={action[:160]!r}")


if __name__ == "__main__":
    main()
