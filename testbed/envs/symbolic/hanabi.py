"""Hanabi — cooperative card game with imperfect self-information.

Each player holds 4–5 cards visible to others but NOT to themselves.
On each turn the active player does one of:
  PLAY <pos>                       — attempt to extend a fireworks pile
  DISCARD <pos>                    — discard to gain a clue token
  HINT <player> COLOR <color>      — tell a player which cards share a color
  HINT <player> RANK  <rank>       — tell a player which cards share a rank

Clue tokens: start at 8, spend 1 per hint, gain 1 per successful rank-5 play or discard.
Fuse tokens: start at 3, lose 1 per illegal play; reach 0 → game ends (score 0 effective).
Score: sum of highest cards successfully played across all 5 colors (max 25).

Reference: Bard et al. (2019). The Hanabi Challenge: A New Frontier for AI Research.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, RenderContext, StepResult

COLORS     = ["red", "yellow", "green", "blue", "white"]
RANKS      = [1, 2, 3, 4, 5]
RANK_COUNTS = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}

HAND_SIZE = {2: 5, 3: 5, 4: 4, 5: 4}
MAX_CLUES = 8
MAX_FUSE  = 3


def _build_deck() -> List[Dict[str, Any]]:
    deck = []
    for color in COLORS:
        for rank, count in RANK_COUNTS.items():
            for _ in range(count):
                deck.append({"color": color, "rank": rank})
    return deck


def _fresh_clue_slot() -> Dict[str, Any]:
    return {"colors": set(), "ranks": set(), "not_colors": set(), "not_ranks": set()}


class HanabiAdapter(SymbolicAdapter):
    """Turn-based Hanabi.  pending() always returns exactly the current player."""

    def __init__(self, num_players: int = 3, seed: int = 0) -> None:
        if num_players not in HAND_SIZE:
            raise ValueError(f"num_players must be in {list(HAND_SIZE)}; got {num_players}")
        super().__init__(num_players=num_players, num_rounds=100)
        self._hand_size = HAND_SIZE[num_players]
        self._seed      = seed

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset()
        rng  = random.Random(seed if seed is not None else self._seed)
        deck = _build_deck()
        rng.shuffle(deck)

        hands: Dict[str, List[Dict[str, Any]]] = {}
        for pid in self._ids:
            hands[pid] = [deck.pop() for _ in range(self._hand_size)]

        clue_knowledge: Dict[str, List[Dict[str, Any]]] = {
            pid: [_fresh_clue_slot() for _ in range(self._hand_size)]
            for pid in self._ids
        }

        self.context.extra.update({
            "deck":                   deck,
            "hands":                  hands,
            "clue_knowledge":         clue_knowledge,
            "fireworks":              {c: 0 for c in COLORS},
            "discard_pile":           [],
            "clue_tokens":            MAX_CLUES,
            "fuse_tokens":            MAX_FUSE,
            "current_player_idx":     0,
            "turns_after_deck_empty": 0,
            "score":                  0,
        })

    # ── observation / pending ──────────────────────────────────────────────────

    def _observation(self, agent_id: str) -> RawObs:
        extra = self.context.extra

        others_hands: Dict[str, List[Dict]] = {
            pid: list(hand)
            for pid, hand in extra["hands"].items()
            if pid != agent_id
        }

        return {
            "agent_id":        agent_id,
            "current_player":  self._ids[extra["current_player_idx"]],
            "is_my_turn":      self._ids[extra["current_player_idx"]] == agent_id,
            "hand_size":       self._hand_size,
            "own_hand_size":   len(extra["hands"][agent_id]),
            "own_clues":       extra["clue_knowledge"][agent_id],
            "others_hands":    others_hands,
            "fireworks":       dict(extra["fireworks"]),
            "discard_pile":    list(extra["discard_pile"]),
            "clue_tokens":     extra["clue_tokens"],
            "fuse_tokens":     extra["fuse_tokens"],
            "deck_remaining":  len(extra["deck"]),
            "score":           extra["score"],
            "history":         list(self.context.history),
        }

    def pending(self) -> List[Tuple[str, RawObs]]:
        extra = self.context.extra
        pid   = self._ids[extra["current_player_idx"]]
        return [(pid, self._observation(pid))]

    # ── submit ─────────────────────────────────────────────────────────────────

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        extra = self.context.extra
        pid   = self._ids[extra["current_player_idx"]]
        action = actions.get(pid) or {"type": "discard", "pos": 1}

        rewards = {p: 0.0 for p in self._ids}

        def _draw() -> None:
            if extra["deck"]:
                card = extra["deck"].pop()
                extra["hands"][pid].append(card)
                extra["clue_knowledge"][pid].append(_fresh_clue_slot())

        act_type = action.get("type", "discard") if isinstance(action, dict) else "discard"

        # ── PLAY ──────────────────────────────────────────────────────────────
        if act_type == "play":
            pos = int(action.get("pos", 1)) - 1  # 0-indexed
            pos = max(0, min(pos, len(extra["hands"][pid]) - 1))
            card = extra["hands"][pid].pop(pos)
            extra["clue_knowledge"][pid].pop(pos)

            if extra["fireworks"][card["color"]] == card["rank"] - 1:
                extra["fireworks"][card["color"]] += 1
                extra["score"] += 1
                if card["rank"] == 5:
                    extra["clue_tokens"] = min(extra["clue_tokens"] + 1, MAX_CLUES)
                self.context.history.append({
                    "turn": self.context.round_index,
                    "player": pid, "action": "play",
                    "card": card, "result": "success",
                })
            else:
                extra["discard_pile"].append(card)
                extra["fuse_tokens"] -= 1
                self.context.history.append({
                    "turn": self.context.round_index,
                    "player": pid, "action": "play",
                    "card": card, "result": "misplay",
                })
            _draw()

        # ── DISCARD ───────────────────────────────────────────────────────────
        elif act_type == "discard":
            pos = int(action.get("pos", 1)) - 1
            pos = max(0, min(pos, len(extra["hands"][pid]) - 1))
            card = extra["hands"][pid].pop(pos)
            extra["clue_knowledge"][pid].pop(pos)
            extra["discard_pile"].append(card)
            extra["clue_tokens"] = min(extra["clue_tokens"] + 1, MAX_CLUES)
            self.context.history.append({
                "turn": self.context.round_index,
                "player": pid, "action": "discard", "card": card,
            })
            _draw()

        # ── HINT ──────────────────────────────────────────────────────────────
        elif act_type == "hint":
            target    = action.get("target", "")
            hint_type = action.get("hint_type", "color")
            value     = action.get("value")

            if extra["clue_tokens"] > 0 and target in extra["hands"] and target != pid:
                extra["clue_tokens"] -= 1
                for slot_idx, card in enumerate(extra["hands"][target]):
                    slot = extra["clue_knowledge"][target][slot_idx]
                    if hint_type == "color":
                        if card["color"] == value:
                            slot["colors"].add(value)
                        else:
                            slot["not_colors"].add(value)
                    else:
                        if card["rank"] == value:
                            slot["ranks"].add(value)
                        else:
                            slot["not_ranks"].add(value)
                self.context.history.append({
                    "turn": self.context.round_index,
                    "player": pid, "action": "hint",
                    "target": target, "hint_type": hint_type, "value": value,
                })

        # ── advance turn ──────────────────────────────────────────────────────
        if len(extra["deck"]) == 0:
            extra["turns_after_deck_empty"] += 1

        extra["current_player_idx"] = (extra["current_player_idx"] + 1) % self.num_players
        self.context.round_index   += 1

        done = (
            extra["fuse_tokens"] <= 0
            or extra["score"] >= 25
            or (len(extra["deck"]) == 0
                and extra["turns_after_deck_empty"] >= self.num_players)
        )
        if done:
            rewards = {p: float(extra["score"]) for p in self._ids}

        return StepResult(
            rewards=rewards,
            done=done,
            info={
                "score":          extra["score"],
                "fuse_tokens":    extra["fuse_tokens"],
                "clue_tokens":    extra["clue_tokens"],
                "deck_remaining": len(extra["deck"]),
            },
        )

    def close(self) -> Dict[str, float]:
        return dict(self.context.last_rewards)
