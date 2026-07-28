"""Parser for Overcooked actions."""
from __future__ import annotations

import re

from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext

_MOVE_RE     = re.compile(r"\bMOVE\s+([NSEW])\b",  re.IGNORECASE)
_PICKUP_RE   = re.compile(r"\bPICK[_\s]?UP\b",      re.IGNORECASE)
_DROP_RE     = re.compile(r"\bDROP\b",              re.IGNORECASE)
_INTERACT_RE = re.compile(r"\bINTERACT\b",          re.IGNORECASE)
_STAY_RE     = re.compile(r"\bSTAY\b",              re.IGNORECASE)

VALID_ACTIONS = {"MOVE N", "MOVE S", "MOVE E", "MOVE W", "PICK_UP", "DROP", "INTERACT", "STAY"}


class OvercookedParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        m = _MOVE_RE.search(completion)
        if m:
            return ParsedAction(value=f"MOVE {m.group(1).upper()}")

        if _PICKUP_RE.search(completion):
            return ParsedAction(value="PICK_UP")

        if _DROP_RE.search(completion):
            return ParsedAction(value="DROP")

        if _INTERACT_RE.search(completion):
            return ParsedAction(value="INTERACT")

        if _STAY_RE.search(completion):
            return ParsedAction(value="STAY")

        return ParseError(
            "Could not parse your action. Respond with exactly one of:\n"
            "  MOVE N | MOVE S | MOVE E | MOVE W\n"
            "  PICK_UP | DROP | INTERACT | STAY"
        )
