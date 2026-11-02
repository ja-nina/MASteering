"""Does one agent's answer reach the other agents' prompts? Checked on the logs, per game.

For every turn whose answer is long enough, a snippet from the START of the answer (where agents reason, before
the move) is searched for in the later prompts of the OTHER agents of the same episode. A game where reasoning
leaks shows a high share; a game that forwards only the parsed move/statement shows ~0 (unless the answer
starts with the statement itself, which the example column shows).

    python scripts/check_leak.py $L/main --match __none --episodes 2
    python scripts/check_leak.py $L/main/poker_4p --match __none -v      # also print one leaked example per game
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def check_episode(path: Path):
    recs = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"parse_ok"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs.append((str(r.get("agent_id")) + (f" (turn {r.get('turn')})"), norm(r.get("completion")),
                         norm((r.get("system_prompt") or "") + " " + (r.get("user_prompt") or ""))))
    checked = leaked = whole = 0
    example = None
    for i, (aid, comp, _) in enumerate(recs):
        if len(comp) < 160:
            continue
        snippet = comp[10:90]  # the start: reasoning comes first; and below the end (comp[-80:]): the move
        me = aid.split(" (")[0]
        others = [(a, p) for a, _, p in recs[i + 1:] if a.split(" (")[0] != me]
        if not others:
            continue
        checked += 1
        hit = [(a, p) for a, p in others if snippet in p]
        if hit:
            leaked += 1
            w = [(a, p) for a, p in hit if comp[-80:] in p]  # the end of the answer too: the whole answer was forwarded
            whole += bool(w)
            if example is None or (w and not example[3]):
                a2, p2 = (w or hit)[0]
                k = p2.find(snippet)
                example = (aid, comp, f"{a2}'s prompt: ...{p2[max(0, k - 120):k + 200]}...", bool(w))
    return checked, leaked, whole, example


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--match", default="__none")
    ap.add_argument("--episodes", type=int, default=2, help="episodes per run")
    ap.add_argument("-v", action="store_true", help="per game, one answer and where it shows up in another prompt")
    ap.add_argument("--chars", type=int, default=400, help="characters of the answer shown with -v")
    a = ap.parse_args(argv)
    runs = sorted({p.parent for r in a.roots for p in Path(r).rglob("episode_*.jsonl") if a.match in str(p.parent.name)})
    games = defaultdict(lambda: [0, 0, 0, None])
    for run in runs:
        for f in sorted(run.glob("episode_*.jsonl"))[:a.episodes]:
            c, l, w, ex = check_episode(f)
            g = games[run.parent.name + (" (image)" if "obs-image" in run.name else "")]
            g[0] += c
            g[1] += l
            g[2] += w
            g[3] = g[3] or ex
    pct = lambda k, n: f"{100 * k / n:5.1f}%" if n else "    –"
    print(f"{'game':34s} {'answers checked':>16s} {'start reaches others':>21s} {'whole answer does':>18s}")
    for game, (c, l, w, ex) in sorted(games.items(), key=lambda kv: -(kv[1][2] / kv[1][0] if kv[1][0] else 0)):
        print(f"{game:34s} {c:16d} {pct(l, c):>21s} {pct(w, c):>18s}")
        if a.v and ex:
            print(f"    {ex[0]} answered ({len(ex[1])} chars): \"{ex[1][:a.chars]}{' ...' if len(ex[1]) > a.chars else ''}\"")
            print(f"    -> {ex[2]}\n")


if __name__ == "__main__":
    main()
