"""Parser for Werewolf actions."""
from __future__ import annotations

import re

from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext

_KILL_RE      = re.compile(r"KILL\s*:\s*(player_\d+)",        re.IGNORECASE)
_INVEST_RE    = re.compile(r"INVESTIGATE\s*:\s*(player_\d+)", re.IGNORECASE)
_SAVE_RE      = re.compile(r"SAVE\s*:\s*(player_\d+)",        re.IGNORECASE)
_VOTE_RE      = re.compile(r"VOTE\s*:\s*(player_\d+)",        re.IGNORECASE)
_STATEMENT_RE = re.compile(r"STATEMENT\s*:\s*(.+)",           re.IGNORECASE | re.DOTALL)
_PLAYER_RE    = re.compile(r"player_\d+",                     re.IGNORECASE)


def _first_valid(text: str, living: list[str]) -> str | None:
    for m in _PLAYER_RE.finditer(text):
        pid = m.group(0)
        if pid.lower() in [l.lower() for l in living]:
            return pid
    return None


class WerewolfParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        phase  = raw_obs.get("phase", "")
        living = raw_obs.get("living", [])

        if phase == "night_werewolf":
            m = _KILL_RE.search(completion)
            if m:
                target = m.group(1)
                if target in living:
                    return ParsedAction(value=target)
            target = _first_valid(completion, living)
            if target:
                return ParsedAction(value=target)
            return ParseError(
                f"Could not find a valid kill target. "
                f"Respond with: KILL: player_X  (living: {', '.join(living)})"
            )

        if phase == "night_seer":
            m = _INVEST_RE.search(completion)
            if m:
                target = m.group(1)
                if target in living and target != agent_id:
                    return ParsedAction(value=target)
            target = _first_valid(completion, [p for p in living if p != agent_id])
            if target:
                return ParsedAction(value=target)
            return ParseError(
                f"Could not find a valid investigation target. "
                f"Respond with: INVESTIGATE: player_X  (living, not yourself)"
            )

        if phase == "night_doctor":
            m = _SAVE_RE.search(completion)
            if m:
                target = m.group(1)
                if target in living:
                    return ParsedAction(value=target)
            target = _first_valid(completion, living)
            if target:
                return ParsedAction(value=target)
            return ParseError(
                f"Could not find a valid save target. "
                f"Respond with: SAVE: player_X  (can include yourself)"
            )

        if phase == "day_discussion":
            m = _STATEMENT_RE.search(completion)
            text = m.group(1).strip() if m else completion.strip()
            if not text:
                return ParseError("Your statement appears empty. Respond with: STATEMENT: <your message>")
            return ParsedAction(value={"statement": text[:500]})   # cap length

        if phase == "day_vote":
            m = _VOTE_RE.search(completion)
            if m:
                target = m.group(1)
                if target in living and target != agent_id:
                    return ParsedAction(value=target)
            target = _first_valid(completion, [p for p in living if p != agent_id])
            if target:
                return ParsedAction(value=target)
            return ParseError(
                f"Could not find a valid vote target. "
                f"Respond with: VOTE: player_X  (living, not yourself)"
            )

        return ParseError(f"Unknown phase: {phase!r}")
