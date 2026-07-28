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

# Number of players who go on each quest (index = quest_number - 1)
# Standard Resistance: Avalon quest sizes by player count.
_QUEST_TEAM_SIZES: Dict[int, List[int]] = {
    5:  [2, 3, 2, 3, 3],
    6:  [2, 3, 4, 3, 4],
    7:  [2, 3, 3, 4, 4],
    8:  [3, 4, 4, 5, 5],
    9:  [3, 4, 4, 5, 5],
    10: [3, 4, 4, 5, 5],
}

# Number of evil players needed to fail a quest (usually 1, but quest 4 at 7+ players needs 2)
_QUEST_FAIL_REQUIRED: Dict[int, List[int]] = {
    5:  [1, 1, 1, 1, 1],
    6:  [1, 1, 1, 1, 1],
    7:  [1, 1, 1, 2, 1],
    8:  [1, 1, 1, 2, 1],
    9:  [1, 1, 1, 2, 1],
    10: [1, 1, 1, 2, 1],
}

# Evil player counts per total player count
_EVIL_COUNTS = {5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4}


class AvalonAdapter(SymbolicAdapter):
    """The Resistance: Avalon with Merlin + optional Percival / Morgana roles.

    Role assignment (seeded):
      Good: Merlin, [Percival], and Loyal Servants to fill.
      Evil: [Mordred], [Morgana], [Oberon], Spies to fill.

    Merlin sees all evil players except Mordred.
    Percival sees Merlin and Morgana (indistinguishable).
    Mordred is hidden from Merlin.
    Oberon is evil but unknown to other evil players.

    TODO: implement _observation() and submit() game logic.
    """

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
        # num_rounds here = max number of leadership rotations; 5 quests × 5 reject limit = 25
        super().__init__(num_players=num_players, num_rounds=25)

        self.include_merlin   = include_merlin
        self.include_percival = include_percival
        self.include_mordred  = include_mordred
        self.include_morgana  = include_morgana
        self.include_oberon   = include_oberon
        self._seed = seed

        self._quest_sizes  = _QUEST_TEAM_SIZES[num_players]
        self._fail_required = _QUEST_FAIL_REQUIRED[num_players]
        self._n_evil       = _EVIL_COUNTS[num_players]

        # Assigned at reset()
        self._roles:        Dict[str, str]  = {}   # player_id → role name
        self._evil_players: List[str]       = []
        self._good_players: List[str]       = []

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset()
        rng = random.Random(seed if seed is not None else self._seed)

        # Build role pool
        evil_pool  = []
        good_pool  = []
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
        self._roles = {pid: role for pid, role in zip(self._ids, combined)}
        self._evil_players = [pid for pid, r in self._roles.items() if r in self.EVIL_ROLES]
        self._good_players = [pid for pid, r in self._roles.items() if r in self.GOOD_ROLES]

        # Phase state
        self.context.extra.update({
            "phase":         "propose",
            "leader_idx":    0,
            "reject_streak": 0,   # consecutive rejected proposals; at 5 → evil wins
            "quest_num":     0,   # 0-indexed; current quest
            "quest_wins":    0,   # quests succeeded by good
            "quest_losses":  0,   # quests failed (≥1 FAIL vote)
            "proposed_team": [],  # player_ids nominated for quest
            "living":        list(self._ids),
            # Per-quest history: [{quest, team, team_votes, quest_votes, succeeded}]
            "quest_history": [],
        })

    # ── observation / pending ──────────────────────────────────────────────────

    def _observation(self, agent_id: str) -> RawObs:
        """Return a role-aware snapshot for agent_id.

        TODO: Complete the role-knowledge rules:
          - Merlin sees evil players (except Mordred).
          - Percival sees Merlin + Morgana (indistinguishable).
          - Evil players see each other (except Oberon sees no one).
          - Ordinary Loyal Servants / Spies see no role information.
        """
        extra = self.context.extra
        role  = self._roles[agent_id]

        # What this agent knows about others' roles
        known_evil: List[str] = []
        if role == "Merlin":
            known_evil = [p for p in self._evil_players if self._roles[p] != "Mordred"]
        elif role in ("Spy", "Morgana", "Mordred"):
            known_evil = [p for p in self._evil_players if self._roles[p] != "Oberon"]

        known_merlin_candidates: List[str] = []
        if role == "Percival":
            known_merlin_candidates = [
                p for p in self._ids
                if self._roles[p] in ("Merlin", "Morgana")
            ]

        return {
            "agent_id":     agent_id,
            "role":         role,
            "side":         "evil" if role in self.EVIL_ROLES else "good",
            "known_evil":   known_evil,
            "known_merlin_candidates": known_merlin_candidates,
            "phase":        extra["phase"],
            "leader":       self._ids[extra["leader_idx"]],
            "quest_num":    extra["quest_num"] + 1,   # 1-indexed for prompts
            "team_size":    self._quest_sizes[extra["quest_num"]],
            "proposed_team": extra["proposed_team"],
            "quest_wins":   extra["quest_wins"],
            "quest_losses": extra["quest_losses"],
            "reject_streak": extra["reject_streak"],
            "living":       extra["living"],
            "quest_history": extra["quest_history"],
        }

    def pending(self) -> List[Tuple[str, RawObs]]:
        extra = self.context.extra
        phase = extra["phase"]
        if phase == "propose":
            leader = self._ids[extra["leader_idx"]]
            return [(leader, self._observation(leader))]
        if phase == "team_vote":
            living = extra["living"]
            return [(p, self._observation(p)) for p in living]
        if phase == "quest_vote":
            team = extra["proposed_team"]
            return [(p, self._observation(p)) for p in team]
        if phase == "assassinate":
            assassin = self._evil_players[0]
            return [(assassin, self._observation(assassin))]
        return []  # done

    # ── submit ─────────────────────────────────────────────────────────────────

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        """Process one phase step and advance to the next phase.

        Expected action formats (see AvalonParser):
          propose:     {"team": ["player_0", "player_2"]}  or comma-separated string
          team_vote:   "APPROVE" | "REJECT"
          quest_vote:  "SUCCESS" | "FAIL"
          assassinate: "player_X"

        TODO: implement the full phase-transition logic.
        Skeleton below shows the state mutations needed.
        """
        extra = self.context.extra
        phase = extra["phase"]
        rewards = {pid: 0.0 for pid in self._ids}

        if phase == "propose":
            # actions = {leader_id: list_of_player_ids or comma_string}
            # TODO: validate team size; store in proposed_team; advance to team_vote
            raise NotImplementedError("propose phase: extract team from action and advance to team_vote")

        if phase == "team_vote":
            # actions = {player_id: "APPROVE" | "REJECT"}
            # TODO: count votes; if majority APPROVE advance to quest_vote;
            #       else increment reject_streak; if streak == 5 evil wins.
            #       Rotate leader.
            raise NotImplementedError("team_vote phase: tally and advance")

        if phase == "quest_vote":
            # actions = {player_id: "SUCCESS" | "FAIL"}
            # TODO: count FAIL votes; if >= _fail_required[quest_num] quest fails.
            #       Advance quest_num; check win/loss (3 wins or 3 losses).
            #       If good wins 3 quests and include_merlin: advance to assassinate.
            raise NotImplementedError("quest_vote phase: tally fails and advance")

        if phase == "assassinate":
            # actions = {assassin_id: "player_X"}
            # TODO: if target == Merlin → evil wins; else good wins.
            raise NotImplementedError("assassinate phase: check target and resolve")

        extra["phase"] = "done"
        done = True
        return StepResult(rewards=rewards, done=done,
                          info={"phase": "done", "winner": "unknown"})

    def agent_ids(self) -> List[str]:
        return list(self._ids)

    def close(self) -> Dict[str, float]:
        return dict(self.context.last_rewards)
