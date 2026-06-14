"""The single, game-agnostic episode loop."""
from __future__ import annotations

from typing import Any, Dict

from testbed.types import ParseError, ParsedAction


class Orchestrator:
    def __init__(self, *, env, renderer, parser, policy, steering, logger,
                 game: str, max_parse_retries: int = 5,
                 fallback_action: Any = 0) -> None:
        self.env = env
        self.renderer = renderer
        self.parser = parser
        self.policy = policy
        self.steering = steering
        self.logger = logger
        self.game = game
        self.max_parse_retries = max_parse_retries
        self.fallback_action = fallback_action

    def _context(self):
        return getattr(self.env, "context", None)

    def _act_one(self, agent_id: str, raw_obs):
        system = self.renderer.system_prompt(agent_id)
        base_user = self.renderer.render(raw_obs, agent_id, self._context())
        system, base_user = self.steering.apply_to_prompt(system, base_user, agent_id)
        spec = self.steering.steering_spec(agent_id)
        spec_id = spec.method if spec is not None else "noop"

        user = base_user
        retries = 0
        completion = ""
        while True:
            completion = self.policy.act(system, user, agent_id, spec)
            result = self.parser.parse(completion, raw_obs, agent_id, self._context())
            if isinstance(result, ParsedAction):
                action = result.value
                break
            if retries >= self.max_parse_retries:
                action = self.fallback_action
                break
            retries += 1
            user = base_user + "\n\n" + result.feedback  # re-prompt with feedback

        return {
            "action": action, "system": system, "user": base_user,
            "completion": completion, "retries": retries, "spec_id": spec_id,
        }

    def run_episode(self) -> Dict[str, float]:
        self.env.reset()
        turn = 0
        last_rewards: Dict[str, float] = {}
        done = False
        while not done:
            pending = self.env.pending()
            actions: Dict[str, Any] = {}
            decided: Dict[str, Dict[str, Any]] = {}
            for agent_id, raw_obs in pending:
                d = self._act_one(agent_id, raw_obs)
                actions[agent_id] = d["action"]
                decided[agent_id] = d

            result = self.env.submit(actions)
            last_rewards = result.rewards
            for agent_id, d in decided.items():
                self.logger.log_step(
                    game=self.game, turn=turn, agent_id=agent_id,
                    system_prompt=d["system"], user_prompt=d["user"],
                    completion=d["completion"], parsed_action=actions[agent_id],
                    parse_retries=d["retries"],
                    reward=result.rewards.get(agent_id, 0.0),
                    steering_spec_id=d["spec_id"],
                )
            turn += 1
            done = result.done

        self.logger.close(summary={"final_rewards": last_rewards})
        return last_rewards
