"""Commons harvest: a common-pool resource game (GovSim fishery style).

A shared stock (e.g. fish in a lake) is harvested by all players each round.
After the harvest the remaining stock regrows (doubles, capped at capacity).
If the stock falls below `collapse_below` the resource collapses and the game
ends early. Each player's reward is what it harvested; the group does best by
harvesting sustainably (at most half the stock in total each round).

Each round has two phases:
  "discussion" — players speak one at a time (`discussion_turns` passes over
                 all living players; 0 disables talk)
  "harvest"    — all players simultaneously state how much they take

Reference: Piatti et al. (2024), Cooperate or Collapse: Emergence of
Sustainable Cooperation in a Society of LLM Agents (GovSim).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, StepResult


class HarvestAdapter(SymbolicAdapter):
    def __init__(self, num_players: int = 5, num_rounds: int = 12,
                 capacity: int = 100, initial_stock: int = 100,
                 collapse_below: int = 5, discussion_turns: int = 1,
                 seed: int = 0) -> None:
        super().__init__(num_players=num_players, num_rounds=num_rounds)
        self.capacity = capacity
        self.initial_stock = initial_stock
        self.collapse_below = collapse_below
        self.discussion_turns = discussion_turns
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.context.extra.update({
            "stock": self.initial_stock,
            "phase": "discussion" if self.discussion_turns else "harvest",
            "speaker": 0,                       # index into the speaking order this round
            "messages": [],                     # this round's messages: {speaker, text}
            "totals": {pid: 0 for pid in self._ids},
        })

    def _order(self) -> List[str]:
        return self._ids * self.discussion_turns

    def _observation(self, agent_id: str) -> RawObs:
        x = self.context.extra
        return {
            "agent_id": agent_id,
            "phase": x["phase"],
            "round_index": self.context.round_index,
            "num_rounds": self.num_rounds,
            "num_players": self.num_players,
            "stock": x["stock"],
            "capacity": self.capacity,
            "collapse_below": self.collapse_below,
            "messages": list(x["messages"]),
            "my_total": x["totals"][agent_id],
            "history": list(self.context.history),
        }

    def pending(self) -> List[Tuple[str, RawObs]]:
        x = self.context.extra
        if x["phase"] == "discussion":
            speaker = self._order()[x["speaker"]]
            return [(speaker, self._observation(speaker))]
        return [(pid, self._observation(pid)) for pid in self._ids]

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        x = self.context.extra
        if x["phase"] == "discussion":
            (speaker, text), = actions.items()
            x["messages"].append({"speaker": speaker, "text": str(text)})
            x["speaker"] += 1
            if x["speaker"] >= len(self._order()):
                x["phase"] = "harvest"
            return StepResult(rewards={pid: 0.0 for pid in self._ids}, done=False,
                              info={"phase": "discussion", "speaker": speaker})
        return self._harvest({pid: max(0, int(actions[pid])) for pid in self._ids})

    def _harvest(self, asked: Dict[str, int]) -> StepResult:
        x = self.context.extra
        stock = x["stock"]
        total_asked = sum(asked.values())
        # more asked than there is: everyone gets the same fraction of their request
        scale = min(1.0, stock / total_asked) if total_asked else 1.0
        taken = {pid: int(a * scale) for pid, a in asked.items()}
        remaining = stock - sum(taken.values())
        collapsed = remaining < self.collapse_below
        regrown = 0 if collapsed else min(self.capacity, 2 * remaining)
        for pid, t in taken.items():
            x["totals"][pid] += t

        self.context.round_index += 1
        self.context.last_rewards = {pid: float(t) for pid, t in taken.items()}
        self.context.history.append({
            "round": self.context.round_index, "stock_before": stock, "asked": asked,
            "taken": taken, "remaining": remaining, "stock_after": regrown,
            "collapsed": collapsed, "messages": x["messages"],
        })
        x.update(stock=regrown, phase="discussion" if self.discussion_turns else "harvest",
                 speaker=0, messages=[])
        done = collapsed or self.context.round_index >= self.num_rounds
        return StepResult(rewards=dict(self.context.last_rewards), done=done,
                          info={"phase": "harvest", "stock_before": stock, "taken": taken,
                                "stock_after": regrown, "collapsed": collapsed,
                                "totals": dict(x["totals"])})
