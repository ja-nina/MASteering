"""Hanabi — cooperative card game with imperfect self-information.

Each player holds 4–5 cards that they can see for other players but NOT for
themselves.  On each turn the active player does one of:
  - PLAY <pos>     — attempt to play one of their own cards onto the fireworks pile
  - DISCARD <pos>  — discard one card to gain a clue token
  - HINT <player> COLOR <color>   — tell a player which of their cards share a color
  - HINT <player> RANK <rank>     — tell a player which of their cards share a rank

Clue tokens: start at 8, spend 1 per hint, gain 1 per discard (max 8).
Fuse tokens:  start at 3, lose 1 per illegal play; reach 0 → game over (score = 0).
Score: number of cards successfully played (max 25 across 5 colors × ranks 1–5).

Reference: Bard et al. (2019). The Hanabi Challenge: A New Frontier for AI Research.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, RenderContext, StepResult

COLORS   = ["red", "yellow", "green", "blue", "white"]
RANKS    = [1, 2, 3, 4, 5]
# Count of each rank in the full deck
RANK_COUNTS = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}

HAND_SIZE = {2: 5, 3: 5, 4: 4, 5: 4}  # cards per player by player count
MAX_CLUES = 8
MAX_FUSE  = 3


def _build_deck() -> List[Dict[str, Any]]:
    deck = []
    for color in COLORS:
        for rank, count in RANK_COUNTS.items():
            for _ in range(count):
                deck.append({"color": color, "rank": rank})
    return deck


class HanabiAdapter(SymbolicAdapter):
    """Turn-based Hanabi.  pending() returns only the current player.

    Observation for the active player includes:
      - Full hands of all OTHER players (color + rank visible)
      - Own hand as a list of "??" entries with accumulated clues
      - Fireworks piles (top rank per color)
      - Discard pile
      - Clue tokens remaining, fuse tokens remaining

    TODO: Implement _observation() and submit() game logic.
    """

    def __init__(self, num_players: int = 3, seed: int = 0) -> None:
        if num_players not in HAND_SIZE:
            raise ValueError(f"num_players must be in {list(HAND_SIZE)}; got {num_players}")
        super().__init__(num_players=num_players, num_rounds=100)  # bounded by deck size
        self._hand_size = HAND_SIZE[num_players]
        self._seed      = seed

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset()
        rng = random.Random(seed if seed is not None else self._seed)
        deck = _build_deck()
        rng.shuffle(deck)

        # Deal hands
        hands: Dict[str, List[Dict[str, Any]]] = {}
        for pid in self._ids:
            hands[pid] = [deck.pop() for _ in range(self._hand_size)]

        # Clue knowledge: for each player, for each card slot, accumulated hints
        # clue_knowledge[pid][slot] = {"colors": set, "ranks": set, "not_colors": set, "not_ranks": set}
        clue_knowledge: Dict[str, List[Dict[str, Any]]] = {}
        for pid in self._ids:
            clue_knowledge[pid] = [
                {"colors": set(), "ranks": set(), "not_colors": set(), "not_ranks": set()}
                for _ in range(self._hand_size)
            ]

        # Fireworks: color → highest rank played so far (0 = none played)
        fireworks = {c: 0 for c in COLORS}

        self.context.extra.update({
            "deck":          deck,
            "hands":         hands,
            "clue_knowledge": clue_knowledge,
            "fireworks":     fireworks,
            "discard_pile":  [],
            "clue_tokens":   MAX_CLUES,
            "fuse_tokens":   MAX_FUSE,
            "current_player_idx": 0,
            "turns_after_deck_empty": 0,  # once deck runs out: each player gets 1 more turn
            "score":         0,
        })

    # ── observation / pending ──────────────────────────────────────────────────

    def _observation(self, agent_id: str) -> RawObs:
        """Return the Hanabi observation for agent_id.

        Key rule: a player can see all hands EXCEPT their own.

        TODO: complete the clue_knowledge representation so that
        the renderer can display accumulated hints clearly.
        """
        extra = self.context.extra
        pid_idx = self._ids.index(agent_id)

        # Other players' hands (fully visible)
        others_hands: Dict[str, List[Dict]] = {
            pid: list(hand)
            for pid, hand in extra["hands"].items()
            if pid != agent_id
        }

        # Own hand: slots with accumulated clue knowledge (not the actual cards)
        own_clues = extra["clue_knowledge"][agent_id]

        return {
            "agent_id":        agent_id,
            "current_player":  self._ids[extra["current_player_idx"]],
            "is_my_turn":      self._ids[extra["current_player_idx"]] == agent_id,
            "hand_size":       self._hand_size,
            "own_hand_size":   len(extra["hands"][agent_id]),
            "own_clues":       own_clues,     # [{colors, ranks, not_colors, not_ranks}]
            "others_hands":    others_hands,  # {player_id: [{color, rank}]}
            "fireworks":       dict(extra["fireworks"]),
            "discard_pile":    list(extra["discard_pile"]),
            "clue_tokens":     extra["clue_tokens"],
            "fuse_tokens":     extra["fuse_tokens"],
            "deck_remaining":  len(extra["deck"]),
            "score":           extra["score"],
            "history":         list(self.context.history),
        }

    def pending(self) -> List[Tuple[str, RawObs]]:
        """Return only the current player."""
        extra = self.context.extra
        pid = self._ids[extra["current_player_idx"]]
        return [(pid, self._observation(pid))]

    # ── submit ─────────────────────────────────────────────────────────────────

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        """Process the current player's action.

        Expected action formats (see HanabiParser):
          PLAY <pos>                        → e.g. "PLAY 2"  (1-indexed)
          DISCARD <pos>                     → e.g. "DISCARD 3"
          HINT <player_id> COLOR <color>    → e.g. "HINT player_1 COLOR red"
          HINT <player_id> RANK <rank>      → e.g. "HINT player_1 RANK 3"

        TODO: implement each branch.
        Pseudocode:
          PLAY:
            card = hands[pid].pop(pos - 1)
            if fireworks[card.color] == card.rank - 1:
                fireworks[card.color] += 1
                score += 1
                if card.rank == 5: clue_tokens = min(clue_tokens + 1, MAX_CLUES)
            else:
                discard_pile.append(card)
                fuse_tokens -= 1
            draw_new_card(pid) if deck else pass
            remove slot from clue_knowledge; append fresh slot

          DISCARD:
            card = hands[pid].pop(pos - 1)
            discard_pile.append(card)
            clue_tokens = min(clue_tokens + 1, MAX_CLUES)
            draw_new_card(pid) if deck else pass

          HINT COLOR/RANK:
            validate clue_tokens > 0 and target != self
            clue_tokens -= 1
            update clue_knowledge[target] for matching / non-matching slots

          After action: advance current_player_idx; check done:
            done if fuse_tokens == 0 or score == 25 or
                   (deck empty and all players have taken a turn since then)
        """
        extra = self.context.extra
        pid   = self._ids[extra["current_player_idx"]]
        action = actions[pid]

        rewards = {p: 0.0 for p in self._ids}

        # TODO: parse and execute action
        raise NotImplementedError(
            f"HanabiAdapter.submit() not implemented. "
            f"Current player: {pid}, action: {action!r}. "
            "See docstring above for the expected logic."
        )

        # After implementing: advance turn
        # extra["current_player_idx"] = (extra["current_player_idx"] + 1) % self.num_players
        # self.context.round_index += 1
        # done = extra["fuse_tokens"] <= 0 or extra["score"] >= 25 or <deck_condition>
        # rewards = {pid: extra["score"] for pid in self._ids}  # shared score
        # return StepResult(rewards=rewards, done=done, info={...})

    def close(self) -> Dict[str, float]:
        return dict(self.context.last_rewards)
