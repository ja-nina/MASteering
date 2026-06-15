"""Goldstone Group Sum game.

Players must collectively reach a hidden target by summing their individual
contributions. After each round every player learns the group sum and the
exact signed error (positive = too high, negative = too low).  Individual
contributions are NOT revealed — only the group total (imperfect monitoring).

Reference: Goldstone et al. (2024). The emergence of specialized roles within
groups. Topics in Cognitive Science, 16(2), 257-281.
"""
from __future__ import annotations

import random
from typing import Dict, Optional

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, StepResult


class GBSAdapter(SymbolicAdapter):
    def __init__(self, num_players: int = 4, num_rounds: int = 10,
                 target: Optional[int] = None,
                 low: int = 20, high: int = 200, seed: int = 0,
                 feedback: str = "exact") -> None:
        """
        feedback:
          "exact"       — agents learn the signed error magnitude each round
                          (e.g. "too HIGH by 23").  Easier to coordinate;
                          rational strategy is to divide error by num_players.
          "directional" — agents only learn the direction, not the magnitude
                          (e.g. "too HIGH").  Harder coordination task; agents
                          must estimate how far off they are from the direction
                          alone, leaving more room for ToM to help.
        """
        super().__init__(num_players=num_players, num_rounds=num_rounds)
        self.low = low
        self.high = high
        if feedback not in ("exact", "directional"):
            raise ValueError(f"feedback must be 'exact' or 'directional', got {feedback!r}")
        self.feedback = feedback
        if target is None:
            target = random.Random(seed).randint(low, high)
        self.target = target

    def _observation(self, agent_id: str) -> RawObs:
        return {
            "agent_id": agent_id,
            "round_index": self.context.round_index,
            "num_players": self.num_players,
            "feedback": self.feedback,
            # history entries expose contributions so the renderer can show
            # each agent its own past submission; other agents' values are
            # filtered out in the renderer (imperfect monitoring).
            "history": list(self.context.history),
        }

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        contributions = {pid: int(actions[pid]) for pid in self._ids}
        group_sum = sum(contributions.values())
        error = group_sum - self.target          # positive = too high

        if error == 0:
            direction = "correct"
        elif error > 0:
            direction = "too_high"
        else:
            direction = "too_low"

        rewards = {pid: 1.0 if error == 0 else 0.0 for pid in self._ids}

        self.context.round_index += 1
        self.context.last_rewards = rewards
        self.context.history.append({
            "round": self.context.round_index,
            "contributions": contributions,
            "group_sum": group_sum,
            "error": error,
            "direction": direction,
        })

        done = direction == "correct" or self.context.round_index >= self.num_rounds
        return StepResult(rewards=rewards, done=done,
                          info={"group_sum": group_sum, "error": error,
                                "direction": direction})
