from __future__ import annotations

from typing import Dict, Optional, Tuple

from testbed.types import SteeringSpec


class PromptInjectionSteering:
    """Per-agent prompt edits.

    per_agent overrides individual agents; default_config applies to every
    agent not explicitly listed (set it to steer all players uniformly).
    Config values: {system_suffix?, user_prefix?}
    """

    def __init__(self, per_agent: Dict[str, Dict[str, str]],
                 default_config: Optional[Dict[str, str]] = None) -> None:
        self.per_agent = per_agent
        self.default_config = default_config or {}

    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> Tuple[str, str]:
        cfg = self.per_agent.get(agent_id) or self.default_config
        if "system_suffix" in cfg:
            system_prompt = system_prompt + cfg["system_suffix"]
        if "user_prefix" in cfg:
            user_prompt = cfg["user_prefix"] + user_prompt
        return system_prompt, user_prompt

    def steering_spec(self, agent_id: str) -> Optional[SteeringSpec]:
        return None  # prompt injection works purely on text
