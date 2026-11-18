from __future__ import annotations

import re

from testbed.parsers.base import keyword_value
from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext

NUMBER = r"\$?\s*-?\d[\d,]*(?:\.\d+)?|\$?\s*-?\.\d+"


class QADebateParser:
    marker = "ANSWER"  # a cut-off answer is finished with 'ANSWER:' (runner.cutoff_feedback)

    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        text = completion or ""
        if raw_obs["options"]:  # multiple choice (mmlu_pro, supergpqa)
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(raw_obs["options"])]
            m = keyword_value(text, "ANSWER", rf"[{letters}{letters.lower()}](?![A-Za-z])")
            if m is None:
                return ParseError(feedback=f"End your reply with 'ANSWER: <letter>', one of {', '.join(letters)}.")
            answer = m.group(1).upper()
        else:
            m = keyword_value(text, "ANSWER", NUMBER)
            if m is None:
                return ParseError(feedback="End your reply with 'ANSWER: <number>' (digits only, no units).")
            answer = re.sub(r"[\s$,]", "", m.group(1))
        return ParsedAction(value={"answer": answer, "text": text[:m.start()].strip()})
