"""SteeringMethod protocol."""
from __future__ import annotations

from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from testbed.types import SteeringSpec


@runtime_checkable
class SteeringMethod(Protocol):
    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> Tuple[str, str]: ...

    def steering_spec(self, agent_id: str) -> Optional[SteeringSpec]: ...

    def on_episode_start(self, env: Any) -> None:
        """Called once after env.reset() before the first step. No-op by default."""
        ...
