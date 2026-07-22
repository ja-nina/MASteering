from __future__ import annotations

from testbed.parsers.base import extract_int
from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class GBSParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        keyword = raw_obs.get("response_keyword", "NUMBER")
        n = extract_int(completion, keyword=keyword)
        if n is None:
            return ParseError(
                feedback=f"I could not find a number. "
                         f"Respond with '{keyword}: <integer>'."
            )
        if n < 0:
            return ParseError(
                feedback=f"Your number must be non-negative, got {n}. "
                         f"Respond with '{keyword}: <integer>'."
            )
        return ParsedAction(value=n)
