"""Near pass-through parser: TextArena validates actions itself."""
from __future__ import annotations

from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class TextArenaParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        text = (completion or "").strip()
        if not text:
            return ParseError(feedback="Empty response. Please provide an action.")
        return ParsedAction(value=text)
