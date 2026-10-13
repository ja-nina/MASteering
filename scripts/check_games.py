"""Set up and reset every game of a sweep on CPU (no model), to catch setup errors before a GPU job.

    python scripts/check_games.py config/sweeps/smoke.yaml

One line per (game, players): OK, or FAIL with the error (unknown env id, wrong player count,
missing package, ...). Exits 1 if any game fails.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testbed.registry import build_game  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # OPENROUTER_API_KEY for TextArena juries


def check(family: str, game_id: str, players: int, env_kwargs: dict) -> str:
    """'' if the game builds and resets, else the error."""
    try:
        env, _, _ = build_game(family=family, game_id=game_id, num_players=players,
                               env_kwargs={"seed": 0, **env_kwargs})
        env.reset()
        close = getattr(env, "close", None)
        if close:
            close()
        return ""
    except Exception as e:  # noqa: BLE001 - report every game, not just the first failure
        return f"{type(e).__name__}: {e}".splitlines()[0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("sweep", help="sweep file, e.g. config/sweeps/main.yaml")
    p.add_argument("--family", help="only games of this family (symbolic | meltingpot | textarena)")
    args = p.parse_args(argv)

    sweep = yaml.safe_load(Path(args.sweep).read_text(encoding="utf-8"))
    failed = 0
    for name, game in sweep["games"].items():
        if args.family and game["family"] != args.family:
            continue
        players = game.get("players", [2])
        for n in players if isinstance(players, list) else [players]:
            error = check(game["family"], game["id"], n, game.get("env_kwargs") or {})
            failed += bool(error)
            print(f"{'FAIL' if error else 'OK  '}  {name} ({game['family']}:{game['id']}, {n}p)"
                  + (f"  {error}" if error else ""), flush=True)
    print(f"{failed} game(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
