from __future__ import annotations

from testbed.parsers.base import extract_int
from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class GBSParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        low, high = raw_obs["low"], raw_obs["high"]
        n = extract_int(completion, keyword="GUESS")
        if n is None:
            return ParseError(
                feedback="I could not find a number. Respond with 'GUESS: <number>'."
            )
        if n < low or n > high:
            return ParseError(
                feedback=(f"Your guess {n} is out of range. Pick an integer between "
                          f"{low} and {high}. Respond with 'GUESS: <number>'.")
            )
        return ParsedAction(value=n)
