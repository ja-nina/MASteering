"""Parser for Werewolf actions."""
from __future__ import annotations

import re

from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext

_KILL_RE      = re.compile(r"KILL\s*:\s*(\S+)",        re.IGNORECASE)
_INVEST_RE    = re.compile(r"INVESTIGATE\s*:\s*(\S+)", re.IGNORECASE)
_SAVE_RE      = re.compile(r"SAVE\s*:\s*(\S+)",        re.IGNORECASE)
_VOTE_RE      = re.compile(r"VOTE\s*:\s*(\S+)",        re.IGNORECASE)
_STATEMENT_RE = re.compile(r"STATEMENT\s*:\s*(.+)",    re.IGNORECASE | re.DOTALL)


def _normalize(name: str, candidates: list[str]) -> str | None:
    """Case-insensitive exact match of name against candidates."""
    nl = name.strip().lower()
    for c in candidates:
        if c.lower() == nl:
            return c
    return None


def _first_valid(text: str, candidates: list[str]) -> str | None:
    """Return the first candidate name found as a whole word in text."""
    lower_map = {c.lower(): c for c in candidates}
    for m in re.finditer(r"\b\w+\b", text):
        hit = lower_map.get(m.group(0).lower())
        if hit is not None:
            return hit
    return None


class WerewolfParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        phase  = raw_obs.get("phase", "")
        living = raw_obs.get("living", [])

        if phase == "night_werewolf":
            m = _KILL_RE.search(completion)
            if m:
                target = _normalize(m.group(1), living)
                if target:
                    return ParsedAction(value=target)
            target = _first_valid(completion, living)
            if target:
                return ParsedAction(value=target)
            return ParseError(
                f"Could not find a valid kill target. "
                f"Respond with: KILL: <name>  (living: {', '.join(living)})"
            )

        if phase == "night_seer":
            others = [p for p in living if p != agent_id]
            m = _INVEST_RE.search(completion)
            if m:
                target = _normalize(m.group(1), others)
                if target:
                    return ParsedAction(value=target)
            target = _first_valid(completion, others)
            if target:
                return ParsedAction(value=target)
            return ParseError(
                f"Could not find a valid investigation target. "
                f"Respond with: INVESTIGATE: <name>  (living, not yourself: {', '.join(others)})"
            )

        if phase == "night_doctor":
            m = _SAVE_RE.search(completion)
            if m:
                target = _normalize(m.group(1), living)
                if target:
                    return ParsedAction(value=target)
            target = _first_valid(completion, living)
            if target:
                return ParsedAction(value=target)
            return ParseError(
                f"Could not find a valid save target. "
                f"Respond with: SAVE: <name>  (can include yourself; living: {', '.join(living)})"
            )

        if phase == "day_discussion":
            m = _STATEMENT_RE.search(completion)
            text = m.group(1).strip() if m else completion.strip()
            if not text:
                return ParseError("Your statement appears empty. Respond with: STATEMENT: <your message>")
            return ParsedAction(value={"statement": text[:500]})

        if phase == "day_vote":
            others = [p for p in living if p != agent_id]
            m = _VOTE_RE.search(completion)
            if m:
                target = _normalize(m.group(1), others)
                if target:
                    return ParsedAction(value=target)
            target = _first_valid(completion, others)
            if target:
                return ParsedAction(value=target)
            return ParseError(
                f"Could not find a valid vote target. "
                f"Respond with: VOTE: <name>  (living, not yourself: {', '.join(others)})"
            )

        return ParseError(f"Unknown phase: {phase!r}")
