"""ActionParser protocol + a shared integer-extraction helper."""
from __future__ import annotations

import re
from typing import Optional, Protocol, runtime_checkable

from testbed.types import ParseResult, RawObs, RenderContext


@runtime_checkable
class ActionParser(Protocol):
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult: ...


def extract_int(text: str, keyword: Optional[str] = None) -> Optional[int]:
    """Extract an integer. If keyword given, only match 'KEYWORD: <n>' — no fallback."""
    if keyword:
        m = re.search(rf"{keyword}\s*[:=]\s*(-?\d+)", text, re.IGNORECASE)
        return int(m.group(1)) if m else None
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None
