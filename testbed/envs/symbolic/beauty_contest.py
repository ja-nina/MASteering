"""Keynesian beauty contest: guess 2/3 of the group average."""
from __future__ import annotations

from typing import Dict

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, StepResult


class BeautyContestAdapter(SymbolicAdapter):
    def __init__(self, num_players: int = 3, num_rounds: int = 5,
                 low: int = 0, high: int = 100) -> None:
        super().__init__(num_players=num_players, num_rounds=num_rounds)
        self.low = low
        self.high = high

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
        choices = {pid: float(actions[pid]) for pid in self._ids}
        mean = sum(choices.values()) / self.num_players
        target = (2.0 / 3.0) * mean

        best = min(abs(v - target) for v in choices.values())
        winners = [pid for pid, v in choices.items() if abs(v - target) == best]
        share = 1.0 / len(winners)
        rewards = {pid: (share if pid in winners else 0.0) for pid in self._ids}

        self.context.round_index += 1
        self.context.last_rewards = rewards
        self.context.history.append({
            "round": self.context.round_index,
            "choices": choices,
            "mean": mean,
            "target": target,
            "winners": winners,
        })
        done = self.context.round_index >= self.num_rounds
        return StepResult(rewards=rewards, done=done,
                          info={"mean": mean, "target": target, "winners": winners})
