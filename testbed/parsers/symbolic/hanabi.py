"""Parser for Hanabi actions."""
from __future__ import annotations

import re

from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext

_PLAY_RE     = re.compile(r"\bPLAY\s+(\d+)\b",                           re.IGNORECASE)
_DISCARD_RE  = re.compile(r"\bDISCARD\s+(\d+)\b",                        re.IGNORECASE)
_HINT_COLOR_RE = re.compile(
    r"\bHINT\s+(player_\d+)\s+COLOR\s+(red|yellow|green|blue|white)\b",  re.IGNORECASE
)
_HINT_RANK_RE  = re.compile(
    r"\bHINT\s+(player_\d+)\s+RANK\s+([1-5])\b",                         re.IGNORECASE
)

VALID_COLORS = {"red", "yellow", "green", "blue", "white"}


class HanabiParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        hand_size    = raw_obs.get("hand_size", 5)
        clue_tokens  = raw_obs.get("clue_tokens", 0)
        others       = list(raw_obs.get("others_hands", {}).keys())

        # PLAY
        m = _PLAY_RE.search(completion)
        if m:
            pos = int(m.group(1))
            if not (1 <= pos <= hand_size):
                return ParseError(
                    f"Card position must be between 1 and {hand_size}; got {pos}. "
                    f"Respond with: PLAY <pos>"
                )
            return ParsedAction(value={"type": "play", "pos": pos})

        # DISCARD
        m = _DISCARD_RE.search(completion)
        if m:
            pos = int(m.group(1))
            if not (1 <= pos <= hand_size):
                return ParseError(
                    f"Card position must be between 1 and {hand_size}; got {pos}. "
                    f"Respond with: DISCARD <pos>"
                )
            return ParsedAction(value={"type": "discard", "pos": pos})

        # HINT COLOR
        m = _HINT_COLOR_RE.search(completion)
        if m:
            target, color = m.group(1), m.group(2).lower()
            if target not in others:
                return ParseError(
                    f"Cannot hint {target}. Valid targets: {', '.join(others)}."
                )
            if clue_tokens <= 0:
                return ParseError("No clue tokens remaining. Choose PLAY or DISCARD instead.")
            return ParsedAction(value={"type": "hint", "target": target, "hint_type": "color", "value": color})

        # HINT RANK
        m = _HINT_RANK_RE.search(completion)
        if m:
            target, rank = m.group(1), int(m.group(2))
            if target not in others:
                return ParseError(
                    f"Cannot hint {target}. Valid targets: {', '.join(others)}."
                )
            if clue_tokens <= 0:
                return ParseError("No clue tokens remaining. Choose PLAY or DISCARD instead.")
            return ParsedAction(value={"type": "hint", "target": target, "hint_type": "rank", "value": rank})

        return ParseError(
            "Could not parse your action. Use one of:\n"
            "  PLAY <pos>\n"
            "  DISCARD <pos>\n"
            "  HINT player_X COLOR <red|yellow|green|blue|white>\n"
            "  HINT player_X RANK <1-5>"
        )
