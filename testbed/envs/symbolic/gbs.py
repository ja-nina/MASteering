"""Group binary search: players collectively converge on a hidden integer."""
from __future__ import annotations

import statistics
from typing import Dict, Optional

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, StepResult


class GBSAdapter(SymbolicAdapter):
    def __init__(self, num_players: int = 3, num_rounds: int = 9,
                 target: Optional[int] = None, low: int = 0, high: int = 100,
                 seed: int = 0) -> None:
        super().__init__(num_players=num_players, num_rounds=num_rounds)
        self.low = low
        self.high = high
        if target is None:
            import random
            target = random.Random(seed).randint(low, high)
        self.target = target

    def _observation(self, agent_id: str) -> RawObs:
        return {
            "agent_id": agent_id,
            "round_index": self.context.round_index,
            "num_players": self.num_players,
            "low": self.low,
            "high": self.high,
            "history": list(self.context.history),
        }

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        guesses = {pid: int(actions[pid]) for pid in self._ids}
        median = int(statistics.median(guesses.values()))
        if median < self.target:
            direction = "higher"   # target is higher than the median
        elif median > self.target:
            direction = "lower"
        else:
            direction = "correct"

        rewards = {pid: (1.0 if g == self.target else 0.0) for pid, g in guesses.items()}

        self.context.round_index += 1
        self.context.last_rewards = rewards
        self.context.history.append({
            "round": self.context.round_index,
            "guesses": guesses,
            "median": median,
            "direction": direction,
        })
        done = direction == "correct" or self.context.round_index >= self.num_rounds
        return StepResult(rewards=rewards, done=done,
                          info={"median": median, "direction": direction})
