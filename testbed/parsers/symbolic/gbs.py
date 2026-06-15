from __future__ import annotations

from testbed.parsers.base import extract_int
from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class GBSParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        n = extract_int(completion, keyword="CONTRIBUTION")
        if n is None:
            return ParseError(
                feedback="I could not find a number. "
                         "Respond with 'CONTRIBUTION: <integer>'."
            )
        if n < 0:
            return ParseError(
                feedback=f"Contributions must be non-negative, got {n}. "
                         "Respond with 'CONTRIBUTION: <integer>'."
            )
        return ParsedAction(value=n)
