"""Werewolf (Mafia) — social deduction with day/night phases.

Players are secretly assigned roles.  Werewolves try to eliminate all Villagers;
Villagers try to identify and eliminate all Werewolves by majority vote.

Episode phases (stored in context.extra["phase"]):
  "night_werewolf" — werewolves choose one player to eliminate (simultaneous)
  "night_seer"     — seer investigates one player's role (if seer is alive)
  "night_doctor"   — doctor saves one player from elimination (if doctor is alive)
  "day_discussion" — one living player speaks per step (round-robin)
  "day_vote"       — all living players vote to eliminate someone (simultaneous)
  "done"           — terminal

Speaker order in day_discussion: pending() advances one step per submit() call.
Discussion has `discussion_rounds_per_day` complete rotations before day_vote.

Roles:
  Villager     — no night action; wins if all werewolves eliminated
  Werewolf     — kills one player per night; wins when werewolves ≥ villagers
  Seer         — investigates one player per night; sees their true role
  Doctor       — saves one player per night (can save self); blocked kill if match
  Hunter       — TODO: shoots one player upon elimination
  Witch        — TODO: one-use kill potion + one-use save potion

Win conditions (checked after each elimination):
  Villagers win: number of werewolves alive == 0
  Werewolves win: number of werewolves >= number of non-werewolves
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, RenderContext, StepResult

ROLE_SIDE: Dict[str, str] = {
    "Villager":  "village",
    "Seer":      "village",
    "Doctor":    "village",
    "Werewolf":  "werewolf",
}

# Default role pools by player count [village_roles, werewolf_count]
DEFAULT_ROLES: Dict[int, Tuple[List[str], int]] = {
    4:  (["Villager", "Seer"],                        1),
    5:  (["Villager", "Villager", "Seer"],             1),
    6:  (["Villager", "Villager", "Seer", "Doctor"],   2),
    7:  (["Villager"] * 3 + ["Seer", "Doctor"],        2),
    8:  (["Villager"] * 4 + ["Seer", "Doctor"],        2),
    9:  (["Villager"] * 4 + ["Seer", "Doctor"],        3),
    10: (["Villager"] * 5 + ["Seer", "Doctor"],        3),
}


class WerewolfAdapter(SymbolicAdapter):
    """Werewolf with configurable roles and per-phase pending().

    Night phases run in order: werewolf → seer → doctor.
    Day phase: discussion (multiple speaking turns) then vote.

    TODO: implement _observation() and submit() game logic.
    """

    def __init__(
        self,
        num_players: int = 6,
        discussion_rounds_per_day: int = 1,
        roles: Optional[List[str]] = None,
        seed: int = 0,
    ) -> None:
        if num_players not in DEFAULT_ROLES and roles is None:
            raise ValueError(f"No default role setup for {num_players} players; pass explicit roles=")
        # num_rounds = generous upper bound on turns across all days
        super().__init__(num_players=num_players, num_rounds=200)
        self._discussion_rounds = discussion_rounds_per_day
        self._seed  = seed
        self._roles_spec = roles  # override; if None use DEFAULT_ROLES

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset()
        rng = random.Random(seed if seed is not None else self._seed)

        # Build role pool
        if self._roles_spec is not None:
            pool = list(self._roles_spec)
            assert len(pool) == self.num_players
        else:
            village_roles, n_wolves = DEFAULT_ROLES[self.num_players]
            pool = list(village_roles) + ["Werewolf"] * n_wolves

        rng.shuffle(pool)
        roles: Dict[str, str] = {pid: role for pid, role in zip(self._ids, pool)}

        werewolves = [pid for pid, r in roles.items() if r == "Werewolf"]
        seer       = next((pid for pid, r in roles.items() if r == "Seer"), None)
        doctor     = next((pid for pid, r in roles.items() if r == "Doctor"), None)

        self.context.extra.update({
            "roles":        roles,           # private; exposed per-role in _observation
            "living":       list(self._ids),
            "werewolves":   werewolves,
            "seer":         seer,
            "doctor":       doctor,
            "phase":        "night_werewolf",
            "day_num":      1,
            # Discussion state
            "speaker_idx":  0,
            "discussion_turn": 0,  # how many speaking turns done this day
            # Night action results (cleared each night)
            "pending_kill": None,   # werewolf target
            "saved":        None,   # doctor's save target
            # Seer's accumulated knowledge
            "seer_knowledge": {},   # {investigated_pid: role}
            # History of eliminations
            "elimination_log": [],  # [{"day": N, "eliminated": pid, "role": role, "by": "vote"|"werewolf"}]
        })

    # ── observation / pending ──────────────────────────────────────────────────

    def _observation(self, agent_id: str) -> RawObs:
        """Return a role-filtered snapshot for agent_id.

        Role visibility:
          Werewolf: knows other werewolves.
          Seer:     knows investigated players' roles + other werewolves if any
                    revealed via the seer_knowledge dict.
          Everyone: knows living players, elimination log, current day.

        TODO: finalize what each role sees and format for the renderer.
        """
        extra = self.context.extra
        role  = extra["roles"][agent_id]

        known_werewolves: List[str] = []
        if role == "Werewolf":
            known_werewolves = [p for p in extra["werewolves"] if p != agent_id]

        seer_knowledge: Dict[str, str] = {}
        if role == "Seer":
            seer_knowledge = dict(extra["seer_knowledge"])

        return {
            "agent_id":         agent_id,
            "role":             role,
            "side":             ROLE_SIDE.get(role, "village"),
            "living":           list(extra["living"]),
            "phase":            extra["phase"],
            "day_num":          extra["day_num"],
            "speaker":          extra["living"][extra["speaker_idx"] % len(extra["living"])]
                                if extra["phase"] == "day_discussion" else None,
            "known_werewolves": known_werewolves,
            "seer_knowledge":   seer_knowledge,
            "elimination_log":  list(extra["elimination_log"]),
            "history":          list(self.context.history),
        }

    def pending(self) -> List[Tuple[str, RawObs]]:
        extra = self.context.extra
        phase = extra["phase"]
        living = extra["living"]

        if phase == "night_werewolf":
            wolves = [p for p in extra["werewolves"] if p in living]
            return [(p, self._observation(p)) for p in wolves]

        if phase == "night_seer":
            seer = extra["seer"]
            if seer and seer in living:
                return [(seer, self._observation(seer))]
            return []  # seer eliminated — skip phase

        if phase == "night_doctor":
            doctor = extra["doctor"]
            if doctor and doctor in living:
                return [(doctor, self._observation(doctor))]
            return []  # doctor eliminated — skip phase

        if phase == "day_discussion":
            idx = extra["speaker_idx"] % len(living)
            speaker = living[idx]
            return [(speaker, self._observation(speaker))]

        if phase == "day_vote":
            return [(p, self._observation(p)) for p in living]

        return []  # done

    # ── submit ─────────────────────────────────────────────────────────────────

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        """Process one phase step.

        Expected action formats (see WerewolfParser):
          night_werewolf:   {wolf_id: "player_X"}  — kill target
          night_seer:       {seer_id: "player_X"}  — investigate target
          night_doctor:     {doctor_id: "player_X"} — save target
          day_discussion:   {speaker_id: "<free text statement>"}
          day_vote:         {player_id: "player_X"} — vote to eliminate

        TODO: implement full phase-transition logic.

        Pseudocode:

        night_werewolf:
          pending_kill = majority_vote(actions.values()) or random if tie
          advance to night_seer (or day_discussion if no seer)

        night_seer:
          seer_knowledge[target] = roles[target]
          advance to night_doctor (or day_discussion if no doctor)

        night_doctor:
          saved = action[doctor_id]
          advance to day_discussion

        [on transition to day_discussion]:
          if pending_kill and pending_kill != saved:
              living.remove(pending_kill)
              elimination_log.append({...})
              check win condition

        day_discussion:
          log the statement; advance speaker_idx
          if discussion_turn >= n_living * _discussion_rounds: advance to day_vote

        day_vote:
          eliminate plurality target; advance to night_werewolf
          check win condition
        """
        extra  = self.context.extra
        phase  = extra["phase"]
        living = extra["living"]
        rewards = {pid: 0.0 for pid in self._ids}

        # --- placeholder transitions (remove TODO raises to activate) ---

        if phase == "night_werewolf":
            raise NotImplementedError("night_werewolf: tally actions and store pending_kill")

        if phase == "night_seer":
            raise NotImplementedError("night_seer: store investigation result in seer_knowledge")

        if phase == "night_doctor":
            raise NotImplementedError("night_doctor: store saved target; resolve pending kill; advance to day")

        if phase == "day_discussion":
            raise NotImplementedError("day_discussion: log statement; advance speaker_idx; check if voting time")

        if phase == "day_vote":
            raise NotImplementedError("day_vote: tally plurality vote; eliminate player; advance to night; check win")

        extra["phase"] = "done"
        return StepResult(rewards=rewards, done=True, info={"winner": "unknown"})

    def close(self) -> Dict[str, float]:
        return dict(self.context.last_rewards)
