"""The Resistance: Avalon — hidden-role social deduction.

N players are secretly assigned roles.  Good roles (Merlin, Loyal Servants)
must succeed 3 of 5 quests; evil roles (Mordred, Spies) must fail 3 quests
or correctly assassinate Merlin at the end.

Episode phases (stored in context.extra["phase"]):
  "propose"      — the current leader nominates a quest team
  "team_vote"    — all living players approve/reject the proposal (simultaneous)
  "quest_vote"   — team members play SUCCESS or FAIL (simultaneous)
  "assassinate"  — evil team identifies Merlin (after good wins 3 quests)
  "done"         — terminal

pending() returns only the agent(s) who should act in the current phase:
  propose     → [leader]
  team_vote   → [all living]
  quest_vote  → [team members]
  assassinate → [assassin (first spy)]
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, RenderContext, StepResult

_QUEST_TEAM_SIZES: Dict[int, List[int]] = {
    5:  [2, 3, 2, 3, 3],
    6:  [2, 3, 4, 3, 4],
    7:  [2, 3, 3, 4, 4],
    8:  [3, 4, 4, 5, 5],
    9:  [3, 4, 4, 5, 5],
    10: [3, 4, 4, 5, 5],
}

# Evil votes needed to fail a quest (quest 4 at 7+ players requires 2)
_QUEST_FAIL_REQUIRED: Dict[int, List[int]] = {
    5:  [1, 1, 1, 1, 1],
    6:  [1, 1, 1, 1, 1],
    7:  [1, 1, 1, 2, 1],
    8:  [1, 1, 1, 2, 1],
    9:  [1, 1, 1, 2, 1],
    10: [1, 1, 1, 2, 1],
}

_EVIL_COUNTS = {5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4}


class AvalonAdapter(SymbolicAdapter):
    """The Resistance: Avalon with Merlin + optional Percival / Morgana / Mordred / Oberon."""

    GOOD_ROLES = ["Merlin", "Percival", "Loyal Servant"]
    EVIL_ROLES = ["Mordred", "Morgana", "Oberon", "Spy"]

    def __init__(
        self,
        num_players: int = 5,
        include_merlin: bool = True,
        include_percival: bool = False,
        include_mordred: bool = False,
        include_morgana: bool = False,
        include_oberon: bool = False,
        seed: int = 0,
    ) -> None:
        if num_players not in _QUEST_TEAM_SIZES:
            raise ValueError(f"num_players must be in {list(_QUEST_TEAM_SIZES)}; got {num_players}")
        super().__init__(num_players=num_players, num_rounds=25)

        self.include_merlin   = include_merlin
        self.include_percival = include_percival
        self.include_mordred  = include_mordred
        self.include_morgana  = include_morgana
        self.include_oberon   = include_oberon
        self._seed = seed

        self._quest_sizes   = _QUEST_TEAM_SIZES[num_players]
        self._fail_required = _QUEST_FAIL_REQUIRED[num_players]
        self._n_evil        = _EVIL_COUNTS[num_players]

        self._roles:        Dict[str, str] = {}
        self._evil_players: List[str]      = []
        self._good_players: List[str]      = []

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset()
        rng = random.Random(seed if seed is not None else self._seed)

        evil_pool: List[str] = []
        good_pool: List[str] = []
        if self.include_mordred:  evil_pool.append("Mordred")
        if self.include_morgana:  evil_pool.append("Morgana")
        if self.include_oberon:   evil_pool.append("Oberon")
        while len(evil_pool) < self._n_evil:
            evil_pool.append("Spy")

        if self.include_merlin:   good_pool.append("Merlin")
        if self.include_percival: good_pool.append("Percival")
        n_good = self.num_players - self._n_evil
        while len(good_pool) < n_good:
            good_pool.append("Loyal Servant")

        combined = good_pool[:n_good] + evil_pool[:self._n_evil]
        rng.shuffle(combined)
        self._roles        = {pid: role for pid, role in zip(self._ids, combined)}
        self._evil_players = [p for p, r in self._roles.items() if r in self.EVIL_ROLES]
        self._good_players = [p for p, r in self._roles.items() if r in self.GOOD_ROLES]

        self.context.extra.update({
            "phase":         "propose",
            "leader_idx":    0,
            "reject_streak": 0,
            "quest_num":     0,
            "quest_wins":    0,
            "quest_losses":  0,
            "proposed_team": [],
            "living":        list(self._ids),
            "quest_history": [],
        })

    # ── observation / pending ──────────────────────────────────────────────────

    def _observation(self, agent_id: str) -> RawObs:
        extra = self.context.extra
        role  = self._roles[agent_id]

        known_evil: List[str] = []
        if role == "Merlin":
            known_evil = [p for p in self._evil_players if self._roles[p] != "Mordred"]
        elif role in ("Spy", "Morgana", "Mordred"):
            known_evil = [p for p in self._evil_players if self._roles[p] != "Oberon"]

        known_merlin_candidates: List[str] = []
        if role == "Percival":
            known_merlin_candidates = [
                p for p in self._ids if self._roles[p] in ("Merlin", "Morgana")
            ]

        return {
            "agent_id":                agent_id,
            "role":                    role,
            "side":                    "evil" if role in self.EVIL_ROLES else "good",
            "known_evil":              known_evil,
            "known_merlin_candidates": known_merlin_candidates,
            "phase":                   extra["phase"],
            "leader":                  self._ids[extra["leader_idx"]],
            "quest_num":               extra["quest_num"] + 1,
            "team_size":               self._quest_sizes[extra["quest_num"]],
            "proposed_team":           extra["proposed_team"],
            "quest_wins":              extra["quest_wins"],
            "quest_losses":            extra["quest_losses"],
            "reject_streak":           extra["reject_streak"],
            "living":                  extra["living"],
            "quest_history":           extra["quest_history"],
        }

    def pending(self) -> List[Tuple[str, RawObs]]:
        extra = self.context.extra
        phase = extra["phase"]
        if phase == "propose":
            leader = self._ids[extra["leader_idx"]]
            return [(leader, self._observation(leader))]
        if phase == "team_vote":
            return [(p, self._observation(p)) for p in extra["living"]]
        if phase == "quest_vote":
            return [(p, self._observation(p)) for p in extra["proposed_team"]]
        if phase == "assassinate":
            assassin = self._evil_players[0]
            return [(assassin, self._observation(assassin))]
        return []

    # ── submit ─────────────────────────────────────────────────────────────────

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        extra   = self.context.extra
        phase   = extra["phase"]
        rewards = {pid: 0.0 for pid in self._ids}
        self.context.round_index += 1

        # ── propose: leader picks the quest team ──────────────────────────────
        if phase == "propose":
            leader = self._ids[extra["leader_idx"]]
            raw    = actions.get(leader, {})
            team: List[str] = raw.get("team", []) if isinstance(raw, dict) else []
            expected = self._quest_sizes[extra["quest_num"]]
            team = [p for p in team if p in extra["living"]]
            # Pad with living players not already on team if too short
            if len(team) < expected:
                fillers = [p for p in extra["living"] if p not in team]
                team += fillers[:expected - len(team)]
            team = team[:expected]
            extra["proposed_team"] = team
            extra["phase"] = "team_vote"
            return StepResult(rewards=rewards, done=False,
                              info={"phase": "team_vote", "team": team})

        # ── team_vote: majority approves or rejects ───────────────────────────
        if phase == "team_vote":
            approves = sum(1 for v in actions.values() if str(v).upper() == "APPROVE")
            rejects  = len(actions) - approves
            if approves > rejects:
                extra["reject_streak"] = 0
                extra["phase"] = "quest_vote"
                return StepResult(rewards=rewards, done=False,
                                  info={"phase": "quest_vote", "team_approved": True})
            else:
                extra["reject_streak"] += 1
                extra["leader_idx"] = (extra["leader_idx"] + 1) % self.num_players
                if extra["reject_streak"] >= 5:
                    rewards = {p: (1.0 if p in self._evil_players else 0.0)
                               for p in self._ids}
                    extra["phase"] = "done"
                    return StepResult(rewards=rewards, done=True,
                                      info={"winner": "evil",
                                            "reason": "5_consecutive_rejections"})
                extra["phase"] = "propose"
                return StepResult(rewards=rewards, done=False,
                                  info={"phase": "propose", "team_approved": False,
                                        "reject_streak": extra["reject_streak"]})

        # ── quest_vote: team members play SUCCESS / FAIL ──────────────────────
        if phase == "quest_vote":
            team       = extra["proposed_team"]
            fail_count = sum(
                1 for pid in team
                if str(actions.get(pid, "SUCCESS")).upper() == "FAIL"
            )
            succeeded = fail_count < self._fail_required[extra["quest_num"]]

            extra["quest_history"].append({
                "quest":      extra["quest_num"] + 1,
                "team":       list(team),
                "fail_votes": fail_count,
                "succeeded":  succeeded,
            })
            if succeeded:
                extra["quest_wins"] += 1
            else:
                extra["quest_losses"] += 1

            extra["quest_num"]    += 1
            extra["leader_idx"]    = (extra["leader_idx"] + 1) % self.num_players
            extra["reject_streak"] = 0
            extra["proposed_team"] = []

            if extra["quest_wins"] >= 3:
                if self.include_merlin:
                    extra["phase"] = "assassinate"
                    return StepResult(rewards=rewards, done=False,
                                      info={"phase": "assassinate"})
                rewards = {p: (1.0 if p in self._good_players else 0.0) for p in self._ids}
                extra["phase"] = "done"
                return StepResult(rewards=rewards, done=True, info={"winner": "good"})

            if extra["quest_losses"] >= 3:
                rewards = {p: (1.0 if p in self._evil_players else 0.0) for p in self._ids}
                extra["phase"] = "done"
                return StepResult(rewards=rewards, done=True,
                                  info={"winner": "evil", "reason": "3_quest_failures"})

            extra["phase"] = "propose"
            return StepResult(rewards=rewards, done=False,
                              info={"phase": "propose", "quest_succeeded": succeeded,
                                    "quest_wins": extra["quest_wins"],
                                    "quest_losses": extra["quest_losses"]})

        # ── assassinate: assassin tries to identify Merlin ────────────────────
        if phase == "assassinate":
            assassin = self._evil_players[0]
            target   = actions.get(assassin, "")
            merlin   = next((p for p, r in self._roles.items() if r == "Merlin"), None)

            if target == merlin:
                rewards = {p: (1.0 if p in self._evil_players else 0.0) for p in self._ids}
                extra["phase"] = "done"
                return StepResult(rewards=rewards, done=True,
                                  info={"winner": "evil", "reason": "merlin_assassinated",
                                        "target": target, "merlin": merlin})
            rewards = {p: (1.0 if p in self._good_players else 0.0) for p in self._ids}
            extra["phase"] = "done"
            return StepResult(rewards=rewards, done=True,
                              info={"winner": "good", "reason": "merlin_saved",
                                    "target": target, "merlin": merlin})

        extra["phase"] = "done"
        return StepResult(rewards=rewards, done=True, info={"winner": "unknown"})

    def agent_ids(self) -> List[str]:
        return list(self._ids)

    def close(self) -> Dict[str, float]:
        return dict(self.context.last_rewards)
