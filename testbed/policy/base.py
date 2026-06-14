"""Policy protocol + a StubPolicy for GPU-free testing."""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, runtime_checkable

from testbed.types import SteeringSpec


@runtime_checkable
class Policy(Protocol):
    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec]) -> str: ...


class StubPolicy:
    """Returns scripted completions per agent; records calls. No model required."""

    def __init__(self, scripted: Dict[str, List[str]], default: str = "CHOICE: 0") -> None:
        self.scripted = {k: list(v) for k, v in scripted.items()}
        self.default = default
        self.calls: List[Dict[str, str]] = []

    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec]) -> str:
        self.calls.append({"agent_id": agent_id, "system": system_prompt,
                           "user": user_prompt})
        queue = self.scripted.get(agent_id)
        if queue:
            return queue.pop(0)
        return self.default
