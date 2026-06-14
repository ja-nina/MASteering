"""Base class for simultaneous-move symbolic games."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from testbed.types import Action, RawObs, RenderContext, StepResult


class SymbolicAdapter:
    """All players act every round (simultaneous turn model)."""

    def __init__(self, num_players: int, num_rounds: int) -> None:
        self.num_players = num_players
        self.num_rounds = num_rounds
        self._ids = [f"player_{i}" for i in range(num_players)]
        self.context = RenderContext()

    def reset(self) -> None:
        self.context = RenderContext()

    def agent_ids(self) -> List[str]:
        return list(self._ids)

    def legal_actions(self, agent_id: str) -> Optional[object]:
        return None

    def pending(self) -> List[Tuple[str, RawObs]]:
        return [(pid, self._observation(pid)) for pid in self._ids]

    def close(self) -> Dict[str, float]:
        return dict(self.context.last_rewards)

    # --- subclass hooks ---
    def _observation(self, agent_id: str) -> RawObs:
        raise NotImplementedError

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        raise NotImplementedError
