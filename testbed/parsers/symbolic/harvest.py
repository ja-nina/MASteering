from __future__ import annotations

import re

from testbed.parsers.base import extract_int
from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class HarvestParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        if raw_obs["phase"] == "discussion":
            m = re.search(r"MESSAGE\s*:\s*(.+)", completion or "", re.IGNORECASE | re.DOTALL)
            text = m.group(1).strip() if m else ""
            if not text:
                return ParseError(feedback="Respond in the form: MESSAGE: <what you say>")
            return ParsedAction(value=text)
        n = extract_int(completion or "", keyword="HARVEST")
        if n is None or n < 0:
            return ParseError(feedback="Respond in the form: HARVEST: <non-negative integer>")
        if n > raw_obs["stock"]:
            return ParseError(feedback=f"The lake only holds {raw_obs['stock']} tons. "
                                       "Respond in the form: HARVEST: <integer>")
        return ParsedAction(value=n)
