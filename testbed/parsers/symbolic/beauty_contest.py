from __future__ import annotations

from testbed.parsers.base import extract_int
from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class BeautyContestParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        low, high = raw_obs["low"], raw_obs["high"]
        n = extract_int(completion, keyword="CHOICE")
        if n is None:
            return ParseError(
                feedback=("I could not find a number in your reply. Respond with "
                          "'CHOICE: <number>'.")
            )
        if n < low or n > high:
            return ParseError(
                feedback=(f"Your choice {n} is out of range. Pick an integer "
                          f"between {low} and {high}. Respond with 'CHOICE: <number>'.")
            )
        return ParsedAction(value=n)
