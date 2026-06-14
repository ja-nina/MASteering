"""Adapter for TextArena's turn-based, text-native multiplayer games."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from testbed.types import Action, RawObs, RenderContext, StepResult


class TextArenaAdapter:
    def __init__(self, env_id: str, num_players: int, _env=None) -> None:
        self.env_id = env_id
        self.num_players = num_players
        self._env = _env  # inject for tests; otherwise built in reset()
        self.context = RenderContext()
        self._current_pid: Optional[int] = None

    def _pid_str(self, pid: int) -> str:
        return f"player_{pid}"

    def reset(self) -> None:
        if self._env is None:
            import textarena as ta
            self._env = ta.make(env_id=self.env_id)
        self._env.reset(num_players=self.num_players)
        self.context = RenderContext()
        self._current_pid = None

    def agent_ids(self) -> List[str]:
        return [self._pid_str(i) for i in range(self.num_players)]

    def legal_actions(self, agent_id: str) -> Optional[object]:
        return None  # TextArena validates actions itself

    def pending(self) -> List[Tuple[str, RawObs]]:
        pid, observation = self._env.get_observation()
        self._current_pid = pid
        return [(self._pid_str(pid), observation)]

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        agent_id = self._pid_str(self._current_pid)
        action = actions[agent_id]
        done, info = self._env.step(action)
        rewards: Dict[str, float] = {}
        if done:
            ta_rewards, close_info = self._env.close()
            rewards = {self._pid_str(k): float(v) for k, v in ta_rewards.items()}
            self.context.last_rewards = rewards
            info = {**(info or {}), "close_info": close_info}
        return StepResult(rewards=rewards, done=done, info=info or {})

    def close(self) -> Dict[str, float]:
        if self.context.last_rewards:
            return dict(self.context.last_rewards)
        ta_rewards, _ = self._env.close()
        return {self._pid_str(k): float(v) for k, v in ta_rewards.items()}
