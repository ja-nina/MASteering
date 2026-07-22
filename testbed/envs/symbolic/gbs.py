"""Goldstone Group Sum game — and the Persona Picking variant.

Players must collectively reach a hidden target by summing their individual
contributions. After each round every player learns the group sum and the
exact signed error (positive = too high, negative = too low).  Individual
contributions are NOT revealed — only the group total (imperfect monitoring).

Reference: Goldstone et al. (2024). The emergence of specialized roles within
groups. Topics in Cognitive Science, 16(2), 257-281.

Picking variant (hide_group_size=True, feedback="directional"):
  Faithfully replicates Riedl (2025, arXiv 2510.05174).  Each agent picks
  from [0, 50]; agents are not told the group size; feedback is directional
  only ("too HIGH" / "too LOW"). Persona strings are sampled from the
  `personas` list at init time (seeded) and stored per-agent so the renderer
  can prepend them to each agent's system prompt.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, StepResult


class GBSAdapter(SymbolicAdapter):
    def __init__(self, num_players: int = 4, num_rounds: int = 10,
                 target: Optional[int] = None,
                 low: Optional[int] = None, high: Optional[int] = None,
                 seed: int = 0, feedback: str = "exact",
                 hide_group_size: bool = False,
                 personas: Optional[List[str]] = None,
                 persona_mode: str = "plain") -> None:
        """
        low / high  — absolute target range.  Defaults to 5*num_players and
                      50*num_players so each player's fair share is always in
                      [5, 50] regardless of group size.

        feedback:
          "exact"       — agents learn the signed error magnitude each round
                          (e.g. "too HIGH by 23").  Easier to coordinate;
                          rational strategy is to divide error by num_players.
          "directional" — agents only learn the direction, not the magnitude
                          (e.g. "too HIGH").  Harder coordination task; agents
                          must estimate how far off they are from the direction
                          alone, leaving more room for ToM to help.

        hide_group_size:
          When True the observation and renderer never reveal N.  Used for the
          Picking / Persona replication where agents are unaware of group size.

        personas / persona_mode:
          persona_mode="plain"   — no persona prefix; standard game prompt.
          persona_mode="persona" — each agent is assigned one string from
                                   `personas` (sampled without replacement,
                                   seeded by `seed`), prepended to its system
                                   prompt by the renderer.
          persona_mode="tom"     — same as "persona" plus a Theory-of-Mind
                                   instruction appended to the system prompt.
        """
        super().__init__(num_players=num_players, num_rounds=num_rounds)
        self.low  = low  if low  is not None else 5  * num_players
        self.high = high if high is not None else 50 * num_players
        if feedback not in ("exact", "directional"):
            raise ValueError(f"feedback must be 'exact' or 'directional', got {feedback!r}")
        self.feedback = feedback
        self.hide_group_size = hide_group_size
        self.persona_mode = persona_mode

        rng = random.Random(seed)
        if target is None:
            target = rng.randint(self.low, self.high)
        self.target = target

        # Assign one persona per agent (without replacement) when in persona/tom mode.
        self.agent_personas: Dict[str, Optional[str]] = {pid: None for pid in self._ids}
        if personas and persona_mode != "plain":
            pool = list(personas)
            rng.shuffle(pool)
            for pid, persona in zip(self._ids, pool):
                self.agent_personas[pid] = persona

    def _observation(self, agent_id: str) -> RawObs:
        return {
            "agent_id": agent_id,
            "round_index": self.context.round_index,
            "num_rounds": self.num_rounds,
            "num_players": self.num_players,
            "feedback": self.feedback,
            "hide_group_size": self.hide_group_size,
            "persona": self.agent_personas.get(agent_id),
            "persona_mode": self.persona_mode,
            # "FINAL GUESS" for picking variant so parser uses the paper's keyword
            "response_keyword": "FINAL GUESS" if self.hide_group_size else "NUMBER",
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
