from __future__ import annotations

import re

from testbed.envs.symbolic.board import column_letter
from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class BoardParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        legal = [tuple(m) for m in raw_obs["legal"]]
        m = re.search(r"MOVE\s*:\s*([A-Za-z])\s*(\d*)", completion or "", re.IGNORECASE)
        if not m:
            return ParseError(feedback=self._format(raw_obs))
        c = ord(m.group(1).upper()) - ord("A")
        if raw_obs["gravity"]:
            move = next(((r, cc) for r, cc in legal if cc == c), None)
        else:
            move = (int(m.group(2)) - 1, c) if m.group(2) else None
        if move not in legal:
            free = ", ".join(sorted({column_letter(cc) for _, cc in legal})) if raw_obs["gravity"] else \
                ", ".join(f"{column_letter(cc)}{r + 1}" for r, cc in legal)
            return ParseError(feedback=f"That move is not possible. Legal moves: {free}. {self._format(raw_obs)}")
        return ParsedAction(value=move)

    @staticmethod
    def _format(obs) -> str:
        return ("Respond in the form: MOVE: <column letter>" if obs["gravity"]
                else "Respond in the form: MOVE: <column letter><row number>")
