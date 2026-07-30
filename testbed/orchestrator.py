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

    def _extract_player_roles(self, last_info: Dict[str, Any]) -> Dict[str, str]:
        """Return {player_id: role_name} from close_info or env attributes.

        Tries all known role-field names used across TextArena games.
        Integer keys (TextArena's native player IDs) are normalised to "player_N".
        Returns an empty dict when no role data is found.
        """
        _ROLE_KEYS = ("roles", "player_roles", "role_assignments",
                      "role_map", "assignments", "player_role")

        def _normalise(raw: dict) -> Dict[str, str]:
            return {
                (f"player_{k}" if isinstance(k, int) else str(k)): str(v)
                for k, v in raw.items()
            }

        # TextArena: roles live inside close_info
        ci = last_info.get("close_info") or {}
        if isinstance(ci, dict):
            for key in _ROLE_KEYS:
                val = ci.get(key)
                if isinstance(val, dict) and val:
                    return _normalise(val)

        # Symbolic envs may expose roles directly on the env or its context
        for obj in (self.env, getattr(self.env, "context", None)):
            if obj is None:
                continue
            for key in _ROLE_KEYS:
                val = getattr(obj, key, None)
                if isinstance(val, dict) and val:
                    return _normalise(val)

        return {}

    # Keys that signal a single-player elimination event in per-step info
    _ELIM_STEP_KEYS = (
        "eliminated", "killed", "voted_out", "night_kill", "lynched",
        "dead", "eliminated_player", "killed_player", "exile",
    )
    # Keys that carry an ordered elimination list in close_info
    _ELIM_CLOSE_KEYS = (
        "elimination_order", "eliminations", "dead_players",
        "eliminated_players", "kill_order", "deaths",
    )

    def _collect_step_eliminations(
        self,
        turn: int,
        info: Dict[str, Any],
        seen: set,
        events: list,
    ) -> None:
        """Append any new elimination event found in a step's info dict."""
        for key in self._ELIM_STEP_KEYS:
            val = info.get(key)
            if val is None:
                continue
            pid = f"player_{val}" if isinstance(val, int) else str(val)
            if pid not in seen:
                seen.add(pid)
                events.append({"player_id": pid, "turn": turn, "type": key})

    def _finalise_eliminations(
        self,
        last_info: Dict[str, Any],
        seen: set,
        events: list,
    ) -> list:
        """Merge close_info elimination lists with any step-level events already recorded."""
        ci = last_info.get("close_info") or {}
        if isinstance(ci, dict):
            for key in self._ELIM_CLOSE_KEYS:
                val = ci.get(key)
                if isinstance(val, list) and val:
                    for pid_raw in val:
                        pid = f"player_{pid_raw}" if isinstance(pid_raw, int) else str(pid_raw)
                        if pid not in seen:
                            seen.add(pid)
                            # turn=-1 means we know they were eliminated but not when
                            events.append({"player_id": pid, "turn": -1, "type": key})
                    break  # first matching key wins
        return events

    def _count_steered(self) -> int:
        """Number of agents that received non-noop steering this episode (0 for baselines)."""
        from testbed.steering.noop import NoOpSteering
        if isinstance(self.steering, NoOpSteering):
            return 0
        agent_ids: list = (
            self.env.agent_ids() if hasattr(self.env, "agent_ids") else []
        )
        return sum(1 for aid in agent_ids if self.steering.steering_spec(aid) is not None)

    def run_episode(self) -> Dict[str, float]:
        self.env.reset()
        turn = 0
        last_rewards: Dict[str, float] = {}
        last_info: Dict[str, Any] = {}
        done = False
        elim_events: list = []
        elim_seen: set = set()

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
            last_info    = result.info
            self._collect_step_eliminations(turn, result.info, elim_seen, elim_events)
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

        self._finalise_eliminations(last_info, elim_seen, elim_events)
        self.logger.close(summary={
            "final_rewards": last_rewards,
            "final_info": last_info,
            "player_roles": self._extract_player_roles(last_info),
            "eliminations": elim_events,
            "steered": self._count_steered(),
        })
        return last_rewards
