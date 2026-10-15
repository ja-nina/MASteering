"""ActionParser protocol + a shared integer-extraction helper."""
from __future__ import annotations

import re
from typing import Optional, Protocol, runtime_checkable

from testbed.types import ParseResult, RawObs, RenderContext


@runtime_checkable
class ActionParser(Protocol):
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult: ...


def keyword_value(text: str, keyword: str, value: str) -> Optional[re.Match]:
    """The LAST 'KEYWORD: <value>' in `text` (an answer that reconsiders ends with its final choice), allowing
    the markdown and brackets models put around it: '**KEYWORD**: 5', 'KEYWORD: **5**', 'KEYWORD: [5]' (the
    form a prompt like 'KEYWORD: [0-50]' invites). Group 1 = the value."""
    pattern = rf"{keyword}\s*[*_]*\s*[:=]\s*[*_\[(<\"'`]*\s*({value})"
    matches = list(re.finditer(pattern, text or "", re.IGNORECASE))
    return matches[-1] if matches else None


def extract_int(text: str, keyword: Optional[str] = None) -> Optional[int]:
    """Extract an integer. If keyword given, only match 'KEYWORD: <n>' (the last one) — no fallback."""
    if keyword:
        m = keyword_value(text, keyword, r"-?\d+(?!\s*-\s*\d)")  # not a range such as [0-50]
        return int(m.group(1)) if m else None
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None
