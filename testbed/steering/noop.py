from __future__ import annotations

from typing import Optional, Tuple

from testbed.types import SteeringSpec


class NoOpSteering:
    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> Tuple[str, str]:
        return system_prompt, user_prompt

    def steering_spec(self, agent_id: str) -> Optional[SteeringSpec]:
        return None
