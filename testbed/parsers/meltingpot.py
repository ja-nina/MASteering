from __future__ import annotations

import re

from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class MeltingPotParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        names = raw_obs["action_names"]
        fmt = f"Respond in the form: ACTIONS: <action>, ... using only: {', '.join(names)}"
        m = re.search(r"ACTIONS?\s*:\s*(.+)", completion or "", re.IGNORECASE)
        if not m:
            return ParseError(feedback=fmt)
        words = [w.strip(" .`*").upper().replace(" ", "_") for w in re.split(r"[,;\n]+", m.group(1)) if w.strip()]
        plan = [w for w in words if w in names][: raw_obs["plan_length"]]
        unknown = [w for w in words[: raw_obs["plan_length"]] if w not in names]
        if not plan or unknown:
            return ParseError(feedback=f"Unknown action(s): {', '.join(unknown) or 'none given'}. {fmt}")
        return ParsedAction(value=plan)
