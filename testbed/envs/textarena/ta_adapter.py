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


# sent for an agent that gave no usable answer: TextArena envs expect text and treat it as an invalid move
NO_ANSWER = "(no answer)"

# role names (lower-cased substrings) of the deceiving side in TextArena's hidden-role games
_EVIL_ROLE_WORDS = ("mafia", "werewolf", "spy", "impostor", "traitor", "evil", "morgana", "mordred", "assassin")


class TextArenaAdapter:
    def __init__(self, env_id: str, num_players: int,
                 _env=None, **env_kwargs: Any) -> None:
        self.env_id    = env_id
        self.num_players = num_players
        # Extract seed before storing; ta.make() doesn't accept it.
        self._seed: Optional[int] = env_kwargs.pop("seed", None)
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
        # Seed Python's global random before TextArena env.reset() so that
        # topic/side/role selection (which use random.choice/shuffle before
        # the seed reaches state.reset) are deterministic and identical across
        # all runs sharing the same episode index.
        if self._seed is not None:
            import random as _random
            _random.seed(self._seed)
        if self._env is None:
            import textarena as ta
            self._env = ta.make(env_id=self.env_id, **self._resolve_env_kwargs())
        self._env.reset(num_players=self.num_players, seed=self._seed)
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
        if not isinstance(action, str):  # the runner's fallback (0) after unparseable answers
            action = NO_ANSWER
        done, info = self._env.step(action)
        rewards: Dict[str, float] = {}
        if done:
            ta_rewards, close_info = self._env.close()
            rewards = {self._pid_str(k): float(v) for k, v in ta_rewards.items()}
            self.context.last_rewards = rewards
            info = {**(info or {}), "close_info": close_info}
        return StepResult(rewards=rewards, done=done, info=info or {})

    def roles(self) -> Dict[str, str]:
        """Return {player_N: role_str} for the current episode.

        Reads `player_roles` from the wrapped TextArena env and normalises
        integer keys to "player_N" strings.  This is intentionally `player_roles`
        (the per-seat assignment dict) rather than `roles` (the role-description
        catalogue that SecretMafia also exposes under that name).

        Returns {} when the game hasn't been reset yet or doesn't expose roles.
        """
        if self._env is None:
            return {}
        # Primary: per-seat assignment dict on the env itself (e.g. SecretMafia)
        pr = getattr(self._env, "player_roles", None)
        if isinstance(pr, dict) and pr:
            # Guard against games that put role-description dicts under player_roles
            first_val = next(iter(pr.values()))
            if not isinstance(first_val, dict):
                return {
                    (f"player_{k}" if isinstance(k, int) else str(k)): str(v)
                    for k, v in pr.items()
                }
        # Fallback: game_state dict stored on a `state` sub-object
        state = getattr(self._env, "state", None)
        if state is not None:
            gs = getattr(state, "game_state", None)
            if isinstance(gs, dict):
                for key in ("player_roles", "role_assignments"):
                    val = gs.get(key)
                    if isinstance(val, dict) and val:
                        first_val = next(iter(val.values()))
                        if not isinstance(first_val, dict):
                            return {
                                (f"player_{k}" if isinstance(k, int) else str(k)): str(v)
                                for k, v in val.items()
                            }
        return {}

    def teams(self) -> Dict[str, str]:
        """{player_N: "good" | "evil"}: evil = the hidden, deceiving side (mafia, werewolves, spies)."""
        return {pid: "evil" if any(w in role.lower() for w in _EVIL_ROLE_WORDS) else "good"
                for pid, role in self.roles().items()}

    def close(self) -> Dict[str, float]:
        if self.context.last_rewards:
            return dict(self.context.last_rewards)
        if self._env is None:
            return {}
        ta_rewards, _ = self._env.close()  # rewards are None for a game cut off before its end
        return {self._pid_str(k): float(v) for k, v in (ta_rewards or {}).items()}
