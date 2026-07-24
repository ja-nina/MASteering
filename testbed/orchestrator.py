"""The single, game-agnostic episode loop."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
        system = self.renderer.system_prompt(agent_id, raw_obs)
        base_user = self.renderer.render(raw_obs, agent_id, self._context())
        system, base_user = self.steering.apply_to_prompt(system, base_user, agent_id)
        spec = self.steering.steering_spec(agent_id)
        spec_id = spec.method if spec is not None else "noop"

        user = base_user
        retries = 0
        completion = ""
        truncated = False
        print(system)
        print(user)
        while True:
            act_result = self.policy.act(system, user, agent_id, spec)
            completion, truncated = act_result if isinstance(act_result, tuple) \
                else (act_result, getattr(self.policy, "_last_truncated", False))
            result = self.parser.parse(completion, raw_obs, agent_id, self._context())
            if isinstance(result, ParsedAction):
                action = result.value
                break
            if retries >= self.max_parse_retries:
                action = self.fallback_action
                break
            retries += 1
            user = base_user + "\n\n" + result.feedback  # re-prompt with feedback
        print(completion)
        return {
            "action": action, "system": system, "user": base_user,
            "completion": completion, "retries": retries, "spec_id": spec_id,
            "truncated": truncated,
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
            if len(pending) > 1:
                with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                    fut_to_id = {
                        pool.submit(self._act_one, aid, obs): aid
                        for aid, obs in pending
                    }
                    for fut in as_completed(fut_to_id):
                        aid = fut_to_id[fut]
                        d = fut.result()
                        actions[aid] = d["action"]
                        decided[aid] = d
            else:
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
                    info=result.info,
                    truncated=d.get("truncated", False),
                )
            turn += 1
            done = result.done

        self.logger.close(summary={"final_rewards": last_rewards})
        return last_rewards
