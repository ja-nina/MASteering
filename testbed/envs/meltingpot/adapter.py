"""Melting Pot substrates as games for LLM agents (image observations).

Melting Pot steps every player simultaneously at a fine time scale (hundreds of steps per
episode), so an agent answers with a short plan: up to `plan_length` primitive actions
(FORWARD, TURN_LEFT, FIRE_ZAP, ...), executed one per step. Only agents whose plan has run
out are asked again; the others keep executing theirs. An episode ends when the substrate
ends it or after `max_steps` steps.

Observation: the agent's own egocentric RGB view (Melting Pot's "RGB"), plus the whole map
("WORLD.RGB", a debug observation) with `global_view: true`. Each step returns the rewards of the primitive
steps it ran (like every game: the runner sums them into the episode score); info["totals"] has the running sum. Melting Pot draws its own randomness, so episodes are
not paired across steering conditions the way the other games are.

Reference: Agapiou et al. (2023), Melting Pot 2.0.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from testbed.envs.meltingpot.backend import make_substrate
from testbed.types import Action, RawObs, RenderContext, StepResult


class MeltingPotAdapter:
    def __init__(self, num_players: int, substrate: str, max_steps: int = 200, plan_length: int = 4,
                 global_view: bool = False, python: Optional[str] = None, seed: int = 0) -> None:
        self.num_players, self.substrate = num_players, substrate
        self.max_steps, self.plan_length, self.global_view = max_steps, plan_length, global_view
        self._python = python
        self._sub = None
        self._ids = [f"player_{i}" for i in range(num_players)]
        self.context = RenderContext()

    def reset(self) -> None:
        if self._sub is None:
            self._sub = make_substrate(self.substrate, self.num_players, self._python)
        self._obs = self._sub.reset()
        self.context = RenderContext()
        self.context.extra.update(step=0, done=False, queues={a: [] for a in self._ids},
                                  totals={a: 0.0 for a in self._ids}, last={a: 0.0 for a in self._ids},
                                  plans={a: [] for a in self._ids})

    def agent_ids(self) -> List[str]:
        return list(self._ids)

    def legal_actions(self, agent_id: str) -> List[str]:
        return list(self._sub.action_names)

    def roles(self) -> Dict[str, str]:
        roles = getattr(self._sub, "roles", None) or []
        return {} if set(roles) <= {"default"} else dict(zip(self._ids, roles))

    def teams(self) -> Dict[str, str]:
        return {}

    def _observation(self, agent_id: str) -> RawObs:
        x, i = self.context.extra, self._ids.index(agent_id)
        obs = self._obs[i]
        return {"agent_id": agent_id, "substrate": self.substrate, "step": x["step"], "max_steps": self.max_steps,
                "rgb": obs["RGB"], "world_rgb": obs.get("WORLD.RGB") if self.global_view else None,
                "scalars": {k: v for k, v in obs.items() if isinstance(v, float)},
                "total_reward": x["totals"][agent_id], "last_reward": x["last"][agent_id],
                "last_plan": list(x["plans"][agent_id]), "action_names": list(self._sub.action_names),
                "plan_length": self.plan_length, "num_players": self.num_players}

    def pending(self) -> List[Tuple[str, RawObs]]:
        x = self.context.extra
        if x["done"]:
            return []
        return [(a, self._observation(a)) for a in self._ids if not x["queues"][a]]

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        x = self.context.extra
        for agent, plan in actions.items():
            plan = [p for p in (plan if isinstance(plan, (list, tuple)) else [plan]) if p in self._sub.action_names]
            x["queues"][agent] = (plan or ["NOOP"])[: self.plan_length]
            x["plans"][agent] = list(x["queues"][agent])
            x["last"][agent] = 0.0
        chunk = {a: 0.0 for a in self._ids}
        steps = 0
        while not x["done"] and all(x["queues"].values()):
            names = [x["queues"][a].pop(0) for a in self._ids]
            self._obs, rewards, last = self._sub.step([self._sub.action_names.index(n) for n in names])
            steps += 1
            x["step"] += 1
            for a, r in zip(self._ids, rewards):
                chunk[a] += r
                x["totals"][a] += r
                x["last"][a] += r
            x["done"] = last or x["step"] >= self.max_steps
        self.context.round_index = x["step"]
        self.context.last_rewards = chunk
        return StepResult(rewards=dict(chunk), done=x["done"],
                          info={"step": x["step"], "steps_run": steps, "chunk_rewards": chunk,
                                "totals": dict(x["totals"])})

    def close(self) -> Dict[str, float]:
        if self._sub is not None:
            self._sub.close()
            self._sub = None
        return dict(self.context.extra.get("totals", {}))
