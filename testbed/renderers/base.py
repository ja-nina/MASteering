"""TextRenderer protocol: turn raw observations into prompt text."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from testbed.types import RawObs, RenderContext


@runtime_checkable
class TextRenderer(Protocol):
    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str: ...

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str: ...
