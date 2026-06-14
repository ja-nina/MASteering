"""Pass-through renderer: TextArena already produces the observation string."""
from __future__ import annotations

from testbed.types import RawObs, RenderContext


class TextArenaRenderer:
    def __init__(self, system_prompt_text: str = "") -> None:
        self._sys = system_prompt_text

    def system_prompt(self, agent_id: str) -> str:
        return self._sys

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        return raw_obs  # already a string
