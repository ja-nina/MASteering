"""Adapter for TextArena's turn-based, text-native multiplayer games."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from testbed.types import Action, RawObs, RenderContext, StepResult


def _make_fixed_jury(model_name: str, jury_size: int = 1):
    """Return a jury_class that always calls one fixed OpenRouter model.

    Self-contained — does not depend on textarena.utils so it works across
    textarena versions. Requires OPENROUTER_API_KEY in the environment.
    """
    import os, random

    SYSTEM = (
        "You are a fair and impartial juror. You will be given a context and a "
        "list of possible options. Please select the single most appropriate option "
        "from the list, responding with only that exact option name "
        "(e.g., 'Affirmative', 'Negative', etc.)."
    )

    class FixedJury:
        def __init__(self, jury_size: int = jury_size,
                     options=("Affirmative", "Negative"), **_):
            self.jury_size = jury_size
            self.options   = list(options)

        def evaluate(self, context: str) -> str:
            from openai import OpenAI
            api_key = (os.environ.get("OPENROUTER_API_KEY")
                       or os.environ.get("OPENAI_API_KEY"))
            if not api_key:
                raise RuntimeError(
                    "Fixed jury requires OPENROUTER_API_KEY or OPENAI_API_KEY "
                    "to be set in the environment."
                )
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            votes = []
            for _ in range(self.jury_size):
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user",   "content": context},
                        ],
                        max_tokens=16,
                        temperature=0.0,
                    )
                    text = resp.choices[0].message.content.strip()
                    matched = next(
                        (o for o in self.options if o.lower() in text.lower()),
                        random.choice(self.options),
                    )
                    votes.append(matched)
                except Exception:
                    votes.append(random.choice(self.options))
            # return normalized vote dict matching OpenRouterJury's interface
            return {o: votes.count(o) / len(votes) for o in self.options}

    FixedJury.__name__ = f"FixedJury[{model_name}]"
    return FixedJury


class TextArenaAdapter:
    def __init__(self, env_id: str, num_players: int,
                 _env=None, **env_kwargs: Any) -> None:
        self.env_id    = env_id
        self.num_players = num_players
        self._env_kwargs = env_kwargs
        self._env = _env  # inject for tests; otherwise built in reset()
        self.context = RenderContext()
        self._current_pid: Optional[int] = None

    def _resolve_env_kwargs(self) -> Dict[str, Any]:
        """Convert YAML-friendly env_kwargs to TextArena constructor kwargs.

        Supported special keys:
          jury_model (str)  — fix every juror to this single OpenRouter model ID
          jury_size  (int)  — number of jurors (default 1 when jury_model is set)
        """
        kwargs = dict(self._env_kwargs)
        jury_model = kwargs.pop("jury_model", None)
        if jury_model:
            jury_size = int(kwargs.pop("jury_size", 1))
            kwargs["jury_class"] = _make_fixed_jury(jury_model, jury_size)
        return kwargs

    def _pid_str(self, pid: int) -> str:
        return f"player_{pid}"

    def reset(self) -> None:
        if self._env is None:
            import textarena as ta
            self._env = ta.make(env_id=self.env_id, **self._resolve_env_kwargs())
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
