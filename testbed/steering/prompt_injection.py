from __future__ import annotations

from typing import Dict, Optional, Tuple

from testbed.types import SteeringSpec


class PromptInjectionSteering:
    """Per-agent prompt edits. Config: {agent_id: {system_suffix?, user_prefix?}}."""

    def __init__(self, per_agent: Dict[str, Dict[str, str]]) -> None:
        self.per_agent = per_agent

    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> Tuple[str, str]:
        cfg = self.per_agent.get(agent_id, {})
        if "system_suffix" in cfg:
            system_prompt = system_prompt + cfg["system_suffix"]
        if "user_prefix" in cfg:
            user_prompt = cfg["user_prefix"] + user_prompt
        return system_prompt, user_prompt

    def steering_spec(self, agent_id: str) -> Optional[SteeringSpec]:
        return None  # prompt injection works purely on text
