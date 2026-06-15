"""Activation steering: add a steering vector into the residual stream via a hook.

The apply-path is fully implemented and unit-tested here on a CPU toy module.
Only the offline derivation of *meaningful* vectors is out of scope; drop a
precomputed .pt/.npy vector and point a SteeringSpec at it.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from testbed.types import SteeringSpec


def make_steering_hook(vector, coefficient: float) -> Callable:
    """Return a forward hook that adds coefficient*vector to a module's output.

    Handles modules whose output is a Tensor or a tuple whose first element is the
    hidden-state Tensor (as in most HF decoder blocks).
    """
    import torch

    vec = vector if hasattr(vector, "to") else torch.tensor(vector)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + coefficient * vec.to(hidden.dtype).to(hidden.device)
            return (hidden,) + tuple(output[1:])
        hidden = output
        return hidden + coefficient * vec.to(hidden.dtype).to(hidden.device)

    return hook


class ActivationSteering:
    """Per-agent activation steering.

    per_agent overrides individual agents; default_config applies to every
    agent not explicitly listed (set it to steer all players uniformly).
    """

    def __init__(self, per_agent: Dict[str, Dict],
                 default_config: Optional[Dict] = None) -> None:
        self.per_agent = per_agent
        self.default_config = default_config
        self._cache: Dict[str, object] = {}

    def _cfg_for(self, agent_id: str) -> Optional[Dict]:
        return self.per_agent.get(agent_id) or self.default_config

    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> Tuple[str, str]:
        return system_prompt, user_prompt  # activation steering does not touch text

    def steering_spec(self, agent_id: str) -> Optional[SteeringSpec]:
        cfg = self._cfg_for(agent_id)
        if cfg is None:
            return None
        return SteeringSpec(
            method="activation",
            layer=cfg["layer"],
            vector_path=cfg["vector_path"],
            coefficient=float(cfg.get("coefficient", 1.0)),
        )

    def load_vector(self, agent_id: str):
        import numpy as np
        import torch

        if agent_id in self._cache:
            return self._cache[agent_id]
        spec = self.steering_spec(agent_id)
        if spec is None:
            raise KeyError(f"No activation steering configured for {agent_id}")
        path = spec.vector_path
        if path.endswith(".npy"):
            vec = torch.tensor(np.load(path))
        else:
            vec = torch.load(path)
        vec = vec.float()
        self._cache[agent_id] = vec
        return vec
