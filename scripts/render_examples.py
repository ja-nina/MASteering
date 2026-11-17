"""Render one image observation per image game, mid-game, with the testbed's own adapters, parsers and renderers
(the pictures image-observation agents see). Moves are scripted, not played by the model.

    python scripts/render_examples.py --out <dir> [--size 448]
"""
import argparse
import itertools
from pathlib import Path

from testbed.registry import build_game

# (file name, game id, players, env kwargs, scripted answers cycled over turns, turns before the picture)
GAMES = [
    ("beauty_contest", "beauty_contest", 3, {}, ["CHOICE: 33", "CHOICE: 22", "CHOICE: 50"], 6),
    ("gbs", "gbs", 3, {}, ["NUMBER: 34", "NUMBER: 30", "NUMBER: 20", "NUMBER: 25", "NUMBER: 22", "NUMBER: 18"], 6),
    ("harvest", "harvest", 5, {}, ["MESSAGE: Let's each catch 10 tons.", "HARVEST: 10", "HARVEST: 8",
                                   "HARVEST: 12", "HARVEST: 9", "HARVEST: 10"], 7),
    ("hanabi", "hanabi", 3, {}, ["HINT player_1 COLOR red", "PLAY 0", "DISCARD 3", "HINT player_0 RANK 1"], 5),
    ("tictactoe", "tictactoe", 2, {}, [(1, 1), (0, 0), (0, 2), (2, 0)], 4),
    ("connect_four", "connect_four", 2, {}, [(None, 3), (None, 3), (None, 2), (None, 4),
                                                                (None, 4), (None, 1)], 6),
    ("gomoku", "gomoku", 2, {}, [(7, 7), (7, 8), (6, 6), (8, 8), (5, 5), (6, 8), (8, 6)], 7),
]


def answer(env, parser, aid, obs, script):
    if isinstance(script, tuple):  # board: (row or None, column) -> a legal cell
        legal = env.legal_actions(aid)
        r, c = script
        return next((m for m in legal if m[1] == c and (r is None or m[0] == r)), legal[0])
    if isinstance(obs, dict) and obs.get("phase") == "discussion" and not script.startswith("MESSAGE"):
        script = "MESSAGE: Let's each catch 10 tons."
    elif isinstance(obs, dict) and obs.get("phase") == "harvest" and script.startswith("MESSAGE"):
        script = "HARVEST: 12"
    res = parser.parse(script, obs, aid, env.context)
    return getattr(res, "value", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, gid, n, kw, scripts, turns in GAMES:
        env, renderer, parser = build_game(family="symbolic", game_id=gid, num_players=n, env_kwargs=dict(kw))
        env.reset()  # the runner resets every episode before its first turn
        it = itertools.cycle(scripts)
        for _ in range(turns):
            pend = env.pending()
            if not pend:
                break
            acts = {aid: answer(env, parser, aid, obs, next(it)) for aid, obs in pend}
            if env.submit(acts).done:
                break
        aid, obs = env.pending()[0]
        imgs = renderer.render_images(obs, aid, env.context)
        for i, im in enumerate(imgs):
            p = out / f"obs_{name}{'' if len(imgs) == 1 else f'_{i}'}.png"
            im.save(p)
            print(p, im.size)


if __name__ == "__main__":
    main()
