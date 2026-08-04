"""Role-aware activation steering.

Selects which agent(s) to steer based on their in-game role, determined after
each env.reset(). This means the steered seat changes episode-to-episode as
the game engine re-assigns roles randomly.

Usage
-----
Configure with method "role_aware_activation" in the steering YAML block:

    steering:
      default: role_aware_activation
      mode: adaptive            # additive | adaptive (default: additive)
      target_roles:             # role names to match (case-insensitive)
        - mafia
      default_config:
        vector_path: ${PERSONA_VECTORS_ROOT}/bf16/opportunistic.pt
        coefficient: 1.25
        # layer: auto-selected from companion .json (no need to specify)

Adaptability across games
-------------------------
Different TextArena games use different role vocabularies. Set target_roles to
the exact role strings that game uses, e.g.:

    SecretMafia-v0  →  ["mafia"]
    Werewolf-v0     →  ["werewolf"]
    Avalon-v0       →  ["evil", "morgana", "mordred"]

Role discovery
--------------
After reset(), on_episode_start() walks the env wrapper chain and looks for a
role dict under any of the standard attribute names:
  roles, player_roles, role_assignments, role_map, assignments, player_role

Integer keys are normalised to "player_N" strings. If no roles are found a
warning is emitted and no agents are steered for that episode.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

from testbed.steering.activation import ActivationSteering
from testbed.types import SteeringSpec


class RoleAwareActivationSteering(ActivationSteering):
    """Activation steering that re-targets agents by role each episode.

    Inherits all vector loading, layer resolution, and hook construction from
    ActivationSteering. Adds per-episode role discovery via on_episode_start().
    """

    _ROLE_KEYS = (
        "roles", "player_roles", "role_assignments",
        "role_map", "assignments", "player_role",
    )

    def __init__(
        self,
        target_roles: List[str],
        default_config: Dict,
        mode: str = "adaptive",
    ) -> None:
        # No static per_agent — active agents are discovered each episode.
        super().__init__(per_agent={}, default_config=default_config, mode=mode)
        self._target_roles = {r.lower() for r in target_roles}
        self._active_agents: set = set()

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def steering_spec(self, agent_id) -> Optional[SteeringSpec]:
        """Return spec only for agents whose role was matched this episode."""
        # Normalise int agent IDs to "player_N" to match what _discover_roles stored.
        normalised = f"player_{agent_id}" if isinstance(agent_id, int) else str(agent_id)
        if normalised not in self._active_agents and agent_id not in self._active_agents:
            return None
        # Delegate to parent using default_config (per_agent is always empty here).
        return super().steering_spec(agent_id)

    def on_episode_start(self, env) -> None:
        """Discover roles and arm the matching agents for this episode."""
        self._active_agents = set()
        roles = self._discover_roles(env)

        if not roles:
            warnings.warn(
                f"RoleAwareActivationSteering: could not discover roles from env "
                f"(tried keys: {self._ROLE_KEYS}). "
                "No agents will be steered this episode.",
                stacklevel=2,
            )
            return

        for agent_id, role in roles.items():
            if role.lower() in self._target_roles:
                self._active_agents.add(agent_id)

        if not self._active_agents:
            warnings.warn(
                f"RoleAwareActivationSteering: no agent matched target roles "
                f"{self._target_roles}. Roles observed: "
                f"{ {aid: r for aid, r in roles.items()} }. "
                "No agents will be steered this episode.",
                stacklevel=2,
            )
        else:
            print(
                f"[RoleAwareSteering] Steering {self._active_agents} "
                f"(matched target roles {self._target_roles})"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_roles(self, env) -> Dict[str, str]:
        """Walk the env wrapper chain and return the first role dict found."""
        for obj in self._env_chain(env):
            for key in self._ROLE_KEYS:
                val = getattr(obj, key, None)
                if isinstance(val, dict) and val:
                    return self._normalise(val)
        return {}

    def _env_chain(self, env):
        """Yield env and successive .env attributes until the chain ends."""
        seen: set = set()
        obj = env
        while obj is not None and id(obj) not in seen:
            seen.add(id(obj))
            yield obj
            obj = getattr(obj, "env", None)

    @staticmethod
    def _normalise(raw: dict) -> Dict[str, str]:
        return {
            (f"player_{k}" if isinstance(k, int) else str(k)): str(v)
            for k, v in raw.items()
        }
