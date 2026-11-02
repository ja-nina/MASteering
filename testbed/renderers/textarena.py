"""Pass-through renderer: TextArena already produces the observation string.

With `marker` (game env_kwargs `answer_marker`), the system prompt tells the agent that only its final
'<marker>: ...' line reaches the game and the other players (TextArenaParser extracts it)."""
from __future__ import annotations

from typing import Optional

from testbed.types import RawObs, RenderContext


def marker_instruction(marker: str) -> str:
    return (f"Answer format: you may think first; that text stays private. End your answer with one line "
            f"'{marker}: <what you send to the game>', exactly in the form the game asks for (a move such as "
            f"[Call], or your message to the other players). Only the text after '{marker}:' is sent to the game, "
            f"and the other players see only that.")


class TextArenaRenderer:
    def __init__(self, system_prompt_text: str = "", marker: Optional[str] = None) -> None:
        self._sys = system_prompt_text
        if marker:
            self._sys = (self._sys + "\n\n" if self._sys else "") + marker_instruction(marker)

    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        return self._sys

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        return raw_obs  # already a string
