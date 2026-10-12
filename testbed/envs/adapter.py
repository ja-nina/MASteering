"""The single environment abstraction unifying simultaneous and turn-based games."""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from testbed.types import Action, RawObs, StepResult


@runtime_checkable
class EnvAdapter(Protocol):
    def reset(self) -> None: ...

    def pending(self) -> List[Tuple[str, RawObs]]:
        """Agents that must act now. ALL for simultaneous games, ONE for turn-based."""
        ...

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        """Apply the pending agents' actions and advance the environment."""
        ...

    def agent_ids(self) -> List[str]: ...

    def legal_actions(self, agent_id: str) -> Optional[object]: ...

    def close(self) -> Dict[str, float]:
        """Terminate and return final rewards / game info."""
        ...

    # Hidden-role games also implement roles() -> {agent_id: role} and
    # teams() -> {agent_id: "good" | "evil"}, which `team:` / `roles:` steering groups use.
