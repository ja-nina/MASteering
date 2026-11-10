"""Debug helper (main_v2 interim report 2): how games end, read from the episode summaries.

    python scripts/debug_outcomes.py <logs>/main_v2

1. Avalon: winner and reason (3 quest failures, 5 rejections, Merlin assassinated / saved) per observation mode,
   unsteered and all conditions: why the evil side wins 85% with text but 62% with images.
2. TextArena games: what the summary keeps of the game's end (final_rewards, final_info) for one episode each,
   to see whether the games' own scores (public goods tokens, poker chips, negotiation values) are logged at all
   or only TextArena's win/lose reward.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])

print("== 1. avalon: who wins and why")
for game_dir in sorted(root.glob("avalon_*")):
    counts = {}
    for s in game_dir.glob("*/episode_*.summary.json"):
        run = s.parent.name
        obs = "image" if "obs-image" in run else "text"
        kind = "unsteered" if "__none" in run else "all runs"
        info = json.loads(s.read_text()).get("final_info") or {}
        key = f"{info.get('winner', '?')}: {info.get('reason', '-')}"
        for k in {kind, "all runs"}:
            counts.setdefault((obs, k), Counter())[key] += 1
    for (obs, kind), c in sorted(counts.items()):
        n = sum(c.values())
        print(f"  {game_dir.name} {obs:5s} {kind:9s} n={n:5d}  " +
              "  ".join(f"{k} {v / n:.0%}" for k, v in c.most_common()))

print("\n== 2. what a TextArena summary keeps of the game's end")
for game in ("public_goods_4p", "negotiation_2p", "poker_4p", "mp_prisoners_dilemma_2p"):
    s = next(iter(sorted((root / game).glob("*__none*/episode_*.summary.json"))), None) or \
        next(iter(sorted((root / game).glob("*/episode_*.summary.json"))), None)
    if s is None:
        print(f"  {game}: no summary")
        continue
    d = json.loads(s.read_text())
    print(f"  {game} ({s.parent.name}/{s.name})")
    print(f"    keys: {sorted(k for k in d if k != 'agent_stats')}")
    print(f"    final_rewards: {d.get('final_rewards')}   total_rewards: {d.get('total_rewards')}")
    print(f"    final_info: {json.dumps(d.get('final_info'), default=str)[:1500]}")
