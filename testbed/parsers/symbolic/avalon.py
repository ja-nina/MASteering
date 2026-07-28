"""Parser for The Resistance: Avalon actions."""
from __future__ import annotations

import re
from typing import List

from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext

_APPROVE_RE  = re.compile(r"\bAPPROVE\b", re.IGNORECASE)
_REJECT_RE   = re.compile(r"\bREJECT\b",  re.IGNORECASE)
_SUCCESS_RE  = re.compile(r"\bSUCCESS\b", re.IGNORECASE)
_FAIL_RE     = re.compile(r"\bFAIL\b",    re.IGNORECASE)
_TEAM_RE     = re.compile(r"TEAM\s*:\s*(.+)", re.IGNORECASE)
_VOTE_RE     = re.compile(r"VOTE\s*:\s*(APPROVE|REJECT)", re.IGNORECASE)
_QUEST_RE    = re.compile(r"QUEST\s*:\s*(SUCCESS|FAIL)", re.IGNORECASE)
_ASSASSIN_RE = re.compile(r"ASSASSINATE\s*:\s*(\w+)", re.IGNORECASE)
_PLAYER_RE   = re.compile(r"player_\d+", re.IGNORECASE)


def _extract_player_ids(text: str, living: List[str]) -> List[str]:
    """Extract player_X ids from free text; validate against living list."""
    found = re.findall(r"player_\d+", text, re.IGNORECASE)
    return [p for p in found if p.lower() in [l.lower() for l in living]]


class AvalonParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        phase  = raw_obs.get("phase", "")
        living = raw_obs.get("living", [])

        if phase == "propose":
            m = _TEAM_RE.search(completion)
            if m:
                ids = _extract_player_ids(m.group(1), living)
            else:
                ids = _extract_player_ids(completion, living)
            if not ids:
                return ParseError(
                    "Could not find player IDs in your team proposal. "
                    "Respond with: TEAM: player_X, player_Y"
                )
            return ParsedAction(value={"team": ids})

        if phase == "team_vote":
            m = _VOTE_RE.search(completion)
            if m:
                return ParsedAction(value=m.group(1).upper())
            if _APPROVE_RE.search(completion):
                return ParsedAction(value="APPROVE")
            if _REJECT_RE.search(completion):
                return ParsedAction(value="REJECT")
            return ParseError(
                "Could not find your vote. Respond with: VOTE: APPROVE  or  VOTE: REJECT"
            )

        if phase == "quest_vote":
            m = _QUEST_RE.search(completion)
            if m:
                return ParsedAction(value=m.group(1).upper())
            if _SUCCESS_RE.search(completion):
                return ParsedAction(value="SUCCESS")
            if _FAIL_RE.search(completion):
                return ParsedAction(value="FAIL")
            return ParseError(
                "Could not find your quest play. Respond with: QUEST: SUCCESS  or  QUEST: FAIL"
            )

        if phase == "assassinate":
            m = _ASSASSIN_RE.search(completion)
            if m:
                target = m.group(1).lower()
                matches = [p for p in living if p.lower() == target]
                if matches:
                    return ParsedAction(value=matches[0])
            ids = _extract_player_ids(completion, living)
            if ids:
                return ParsedAction(value=ids[0])
            return ParseError(
                f"Could not find a valid target. Respond with: ASSASSINATE: player_X. "
                f"Living players: {', '.join(living)}"
            )

        return ParseError(f"Unknown phase: {phase!r}")
