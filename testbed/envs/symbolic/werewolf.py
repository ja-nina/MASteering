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

Win conditions (checked after each elimination):
  Villagers win: no werewolves remain
  Werewolves win: werewolves >= non-werewolves among living players
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, RenderContext, StepResult

ROLE_SIDE: Dict[str, str] = {
    "Villager": "village",
    "Seer":     "village",
    "Doctor":   "village",
    "Werewolf": "werewolf",
}

DEFAULT_ROLES: Dict[int, Tuple[List[str], int]] = {
    4:  (["Villager", "Seer"],                          1),
    5:  (["Villager", "Villager", "Seer"],               1),
    6:  (["Villager", "Villager", "Seer", "Doctor"],     2),
    7:  (["Villager"] * 3 + ["Seer", "Doctor"],          2),
    8:  (["Villager"] * 4 + ["Seer", "Doctor"],          2),
    9:  (["Villager"] * 4 + ["Seer", "Doctor"],          3),
    10: (["Villager"] * 5 + ["Seer", "Doctor"],          3),
}


class WerewolfAdapter(SymbolicAdapter):
    """Werewolf with configurable roles and per-phase pending()."""

    def __init__(
        self,
        num_players: int = 6,
        discussion_rounds_per_day: int = 1,
        roles: Optional[List[str]] = None,
        seed: int = 0,
    ) -> None:
        if num_players not in DEFAULT_ROLES and roles is None:
            raise ValueError(
                f"No default role setup for {num_players} players; pass explicit roles="
            )
        super().__init__(num_players=num_players, num_rounds=200)
        self._discussion_rounds = discussion_rounds_per_day
        self._seed        = seed
        self._roles_spec  = roles
        self._rng: random.Random = random.Random(seed)

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset()
        effective = seed if seed is not None else self._seed
        self._rng = random.Random(effective)

        if self._roles_spec is not None:
            pool = list(self._roles_spec)
            assert len(pool) == self.num_players
        else:
            village_roles, n_wolves = DEFAULT_ROLES[self.num_players]
            pool = list(village_roles) + ["Werewolf"] * n_wolves

        self._rng.shuffle(pool)
        roles: Dict[str, str] = {pid: role for pid, role in zip(self._ids, pool)}

        werewolves = [pid for pid, r in roles.items() if r == "Werewolf"]
        seer       = next((pid for pid, r in roles.items() if r == "Seer"),   None)
        doctor     = next((pid for pid, r in roles.items() if r == "Doctor"), None)

        self.context.extra.update({
            "roles":           roles,
            "living":          list(self._ids),
            "werewolves":      werewolves,
            "seer":            seer,
            "doctor":          doctor,
            "phase":           "night_werewolf",
            "day_num":         1,
            "speaker_idx":     0,
            "discussion_turn": 0,
            "pending_kill":    None,
            "saved":           None,
            "seer_knowledge":  {},
            "elimination_log": [],
        })

    # ── helpers ────────────────────────────────────────────────────────────────

    def _resolve_night(self, extra: Dict) -> None:
        """Apply the werewolf kill unless the doctor saved the target."""
        kill   = extra["pending_kill"]
        saved  = extra["saved"]
        living = extra["living"]

        if kill and kill in living and kill != saved:
            living.remove(kill)
            role = extra["roles"][kill]
            extra["elimination_log"].append({
                "day":        extra["day_num"],
                "eliminated": kill,
                "role":       role,
                "by":         "werewolf",
            })
            if kill in extra["werewolves"]:
                extra["werewolves"].remove(kill)
            if kill == extra["seer"]:
                extra["seer"] = None
            if kill == extra["doctor"]:
                extra["doctor"] = None

        extra["pending_kill"] = None
        extra["saved"]        = None

    def _start_day(self, extra: Dict) -> None:
        extra["day_num"]         += 1
        extra["speaker_idx"]      = 0
        extra["discussion_turn"]  = 0
        extra["phase"]            = "day_discussion"

    def _check_win(self, extra: Dict) -> Tuple[bool, Optional[Dict], Optional[Dict]]:
        """Returns (done, rewards_or_None, info_or_None)."""
        living   = extra["living"]
        n_wolves = sum(1 for p in living if p in extra["werewolves"])
        n_others = len(living) - n_wolves

        if n_wolves == 0:
            rewards = {p: (0.0 if extra["roles"].get(p) == "Werewolf" else 1.0)
                       for p in self._ids}
            extra["phase"] = "done"
            return True, rewards, {"winner": "village"}

        if n_wolves >= n_others:
            wolves  = set(extra["werewolves"])
            rewards = {p: (1.0 if p in wolves else 0.0) for p in self._ids}
            extra["phase"] = "done"
            return True, rewards, {"winner": "werewolf"}

        return False, None, None

    # ── observation / pending ──────────────────────────────────────────────────

    def _observation(self, agent_id: str) -> RawObs:
        extra = self.context.extra
        role  = extra["roles"][agent_id]
        living = extra["living"]

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
            "living":           list(living),
            "phase":            extra["phase"],
            "day_num":          extra["day_num"],
            "speaker":          (
                living[extra["speaker_idx"] % len(living)]
                if extra["phase"] == "day_discussion" and living else None
            ),
            "known_werewolves": known_werewolves,
            "seer_knowledge":   seer_knowledge,
            "elimination_log":  list(extra["elimination_log"]),
            "history":          list(self.context.history),
        }

    def pending(self) -> List[Tuple[str, RawObs]]:
        extra  = self.context.extra
        phase  = extra["phase"]
        living = extra["living"]

        if phase == "night_werewolf":
            wolves = [p for p in extra["werewolves"] if p in living]
            return [(p, self._observation(p)) for p in wolves]

        if phase == "night_seer":
            seer = extra["seer"]
            if seer and seer in living:
                return [(seer, self._observation(seer))]
            return []

        if phase == "night_doctor":
            doctor = extra["doctor"]
            if doctor and doctor in living:
                return [(doctor, self._observation(doctor))]
            return []

        if phase == "day_discussion":
            if not living:
                return []
            idx     = extra["speaker_idx"] % len(living)
            speaker = living[idx]
            return [(speaker, self._observation(speaker))]

        if phase == "day_vote":
            return [(p, self._observation(p)) for p in living]

        return []

    # ── submit ─────────────────────────────────────────────────────────────────

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        extra   = self.context.extra
        phase   = extra["phase"]
        living  = extra["living"]
        rewards = {pid: 0.0 for pid in self._ids}
        self.context.round_index += 1

        # ── night_werewolf: wolves vote on a kill target ──────────────────────
        if phase == "night_werewolf":
            non_wolves = [p for p in living if p not in extra["werewolves"]]
            valid = [v for v in actions.values() if v in non_wolves]
            if valid:
                counts     = Counter(valid)
                max_c      = max(counts.values())
                candidates = sorted(p for p, c in counts.items() if c == max_c)
                extra["pending_kill"] = candidates[0]
            elif non_wolves:
                extra["pending_kill"] = self._rng.choice(non_wolves)
            else:
                extra["pending_kill"] = None

            seer   = extra["seer"]
            doctor = extra["doctor"]
            if seer and seer in living:
                extra["phase"] = "night_seer"
            elif doctor and doctor in living:
                extra["phase"] = "night_doctor"
            else:
                self._resolve_night(extra)
                done, win_rewards, info = self._check_win(extra)
                if done:
                    return StepResult(rewards=win_rewards, done=True, info=info)
                self._start_day(extra)
            return StepResult(rewards=rewards, done=False, info={"phase": extra["phase"]})

        # ── night_seer: seer investigates one player ──────────────────────────
        if phase == "night_seer":
            target = next(iter(actions.values()), None) if actions else None
            if target and target in extra["roles"]:
                extra["seer_knowledge"][target] = extra["roles"][target]

            doctor = extra["doctor"]
            if doctor and doctor in living:
                extra["phase"] = "night_doctor"
            else:
                self._resolve_night(extra)
                done, win_rewards, info = self._check_win(extra)
                if done:
                    return StepResult(rewards=win_rewards, done=True, info=info)
                self._start_day(extra)
            return StepResult(rewards=rewards, done=False, info={"phase": extra["phase"]})

        # ── night_doctor: doctor saves one player ─────────────────────────────
        if phase == "night_doctor":
            target = next(iter(actions.values()), None) if actions else None
            extra["saved"] = target if (target and target in living) else None
            self._resolve_night(extra)
            done, win_rewards, info = self._check_win(extra)
            if done:
                return StepResult(rewards=win_rewards, done=True, info=info)
            self._start_day(extra)
            return StepResult(rewards=rewards, done=False, info={"phase": extra["phase"]})

        # ── day_discussion: one speaker makes a statement ─────────────────────
        if phase == "day_discussion":
            if actions:
                speaker = next(iter(actions))
                stmt    = actions[speaker]
                if isinstance(stmt, dict):
                    stmt = stmt.get("statement", "")
                self.context.history.append({
                    "day":       extra["day_num"],
                    "phase":     "discussion",
                    "speaker":   speaker,
                    "statement": str(stmt)[:500],
                })
            extra["speaker_idx"]     += 1
            extra["discussion_turn"] += 1
            total_needed = len(living) * self._discussion_rounds
            if extra["discussion_turn"] >= total_needed:
                extra["phase"] = "day_vote"
            return StepResult(rewards=rewards, done=False, info={"phase": extra["phase"]})

        # ── day_vote: plurality vote to eliminate ─────────────────────────────
        if phase == "day_vote":
            valid_votes = {
                voter: target
                for voter, target in actions.items()
                if target in living and target != voter
            }
            if valid_votes:
                counts     = Counter(valid_votes.values())
                max_c      = max(counts.values())
                candidates = sorted(p for p, c in counts.items() if c == max_c)
                eliminated = candidates[0]

                living.remove(eliminated)
                extra["elimination_log"].append({
                    "day":        extra["day_num"],
                    "eliminated": eliminated,
                    "role":       extra["roles"][eliminated],
                    "by":         "vote",
                })
                if eliminated in extra["werewolves"]:
                    extra["werewolves"].remove(eliminated)
                if eliminated == extra["seer"]:
                    extra["seer"] = None
                if eliminated == extra["doctor"]:
                    extra["doctor"] = None

                done, win_rewards, info = self._check_win(extra)
                if done:
                    return StepResult(rewards=win_rewards, done=True, info=info)

            extra["phase"]        = "night_werewolf"
            extra["pending_kill"] = None
            extra["saved"]        = None
            return StepResult(rewards=rewards, done=False, info={"phase": "night_werewolf"})

        extra["phase"] = "done"
        return StepResult(rewards=rewards, done=True, info={"winner": "unknown"})

    def close(self) -> Dict[str, float]:
        return dict(self.context.last_rewards)
