"""TextArena parser. Default: near pass-through (TextArena validates actions itself), so the whole answer is sent
to the game, and TextArena posts it to the other players.

With `marker` (game env_kwargs `answer_marker: ACTION`, see TextArenaRenderer for the instruction the agents get):
only the text after the LAST 'ACTION:' is sent; everything before it (the agent's reasoning, private cards, plans)
stays private. The rest of the marker's line is the action; if that is empty, the next non-empty line. An answer
without the marker is retried with feedback, as in the other games' parsers.
"""
from __future__ import annotations

import re
from typing import Optional

from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class TextArenaParser:
    def __init__(self, marker: Optional[str] = None) -> None:
        self.marker = marker

    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        text = (completion or "").strip()
        if not text:
            return ParseError(feedback="Empty response. Please provide an action.")
        if not self.marker:
            return ParsedAction(value=text)
        # '**ACTION:**', 'ACTION :', '### ACTION:' ...; the last one counts (an answer that reconsiders ends with it)
        matches = list(re.finditer(rf"{re.escape(self.marker)}\s*[*_]*\s*:\s*[*_]*", text, re.IGNORECASE))
        if not matches:
            return ParseError(feedback=f"End your answer with a line '{self.marker}: <what you send to the game>'. "
                                       f"Only that text reaches the game and the other players.")
        rest = text[matches[-1].end():].strip("\n")
        lines = [ln.strip() for ln in rest.split("\n")]
        action = lines[0] if lines and lines[0] else next((ln for ln in lines if ln), "")
        action = action.strip().strip("*_`").strip()
        if not action:
            return ParseError(feedback=f"Nothing followed '{self.marker}:'. "
                                       f"End your answer with '{self.marker}: <what you send to the game>'.")
        return ParsedAction(value=action)
