"""Lockstep scheduler: many episodes (of one or several runs) share every generate call.

Turn-based games produce one request at a time, which would leave vLLM idle. Instead each
active episode exposes the turns it is waiting for; one scheduler step generates all of them
in one batch, optionally reads them with the monitor, and hands them back. An episode applies
its pending agents' actions once all of them have a valid (or fallback) action, so
simultaneous and turn-based games are handled alike. Parse failures are re-asked in the next
step with the parser's feedback appended to the prompt.
"""
from __future__ import annotations

import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from testbed.engine import Generator, Monitor, Reader, TurnRequest, TurnResult
from testbed.steering import Steer
from testbed.types import ParsedAction
from testbed.util import log, stable_seed
from testbed.visual import fit, modes, observe

# keys that mark a single-player elimination in a step's info / an ordered list in close_info
_ELIM_STEP_KEYS = ("eliminated", "killed", "voted_out", "night_kill", "lynched", "dead",
                   "eliminated_player", "killed_player", "exile")
_ELIM_CLOSE_KEYS = ("elimination_order", "eliminations", "dead_players", "eliminated_players",
                    "kill_order", "deaths")


@dataclass
class Job:
    """One episode to play: everything an EpisodeRun needs, built lazily when it starts."""
    run_index: int
    run_id: str
    game: str
    episode: int
    make_env: Callable[[], tuple]                 # () -> (env, renderer, parser), env already reset
    assign: Callable[[Any], Dict[str, Steer]]     # env -> {agent_id: Steer}
    make_logger: Callable[[], Any]
    max_parse_retries: int = 3
    max_tokens: Optional[int] = None
    max_turns: int = 1000
    fallback_action: Any = 0
    observation: str = "text"                     # text | image | both (testbed/visual.py)
    image_size: int = 448                         # longer side of every image, in pixels


class EpisodeRun:
    def __init__(self, job: Job) -> None:
        self.job = job
        self.env, self.renderer, self.parser = job.make_env()
        if job.observation not in modes(self.renderer):
            raise ValueError(f"{job.game} has no '{job.observation}' observation; it supports {modes(self.renderer)}")
        self.steers = job.assign(self.env)
        self.logger = job.make_logger()
        self.turn = 0
        self.done = False
        self.waiting: Dict[str, dict] = {}     # agent_id -> state of its pending turn
        self.last_rewards: Dict[str, float] = {}
        self.total_rewards: Dict[str, float] = {}  # every step's rewards summed: the episode's score
        self.last_info: Dict[str, Any] = {}
        self.eliminations: List[dict] = []

    def _context(self):
        return getattr(self.env, "context", None)

    def requests(self) -> List[TurnRequest]:
        if not self.waiting and not self.done:
            for agent_id, raw_obs in self.env.pending():
                system = self.renderer.system_prompt(agent_id, raw_obs)
                steer = self.steers.get(agent_id)
                if steer is not None and steer.persona:
                    system = f"{steer.persona}\n\n{system}"
                user, images = observe(self.renderer, self.job.observation, raw_obs, agent_id, self._context())
                self.waiting[agent_id] = {"raw_obs": raw_obs, "system": system, "base_user": user,
                                          "user": user, "attempt": 0, "record": None, "attempts": [],
                                          "images": [fit(im, self.job.image_size) for im in images]}
        return [TurnRequest(key=(self.job.run_index, self.job.episode, self.turn, aid, st["attempt"]),
                            system=st["system"], user=st["user"], max_tokens=self.job.max_tokens,
                            seed=stable_seed(self.job.run_id, self.job.episode, self.turn, aid, st["attempt"]),
                            steer=self.steers.get(aid), images=st["images"])
                for aid, st in self.waiting.items() if st["record"] is None]

    def deliver(self, results: Dict[tuple, TurnResult]) -> List[dict]:
        """Parse this episode's results; returns the turn records that are now final."""
        final = []
        for aid, st in self.waiting.items():
            res = results.get((self.job.run_index, self.job.episode, self.turn, aid, st["attempt"]))
            if res is None or st["record"] is not None:
                continue
            parsed = self.parser.parse(res.text, st["raw_obs"], aid, self._context())
            st["attempts"].append({"attempt": st["attempt"], "user_prompt": st["user"], "completion": res.text,
                                   "finish_reason": res.finish_reason, "n_tokens": res.n_tokens,
                                   "n_prompt_tokens": res.n_prompt_tokens,
                                   "parse_error": None if isinstance(parsed, ParsedAction) else parsed.feedback})
            if isinstance(parsed, ParsedAction) or st["attempt"] >= self.job.max_parse_retries:
                action = parsed.value if isinstance(parsed, ParsedAction) else self.job.fallback_action
                st["record"] = {"agent_id": aid, "action": action, "result": res, "monitor": {},
                                "parse_ok": isinstance(parsed, ParsedAction)}
                final.append(st["record"])
            else:
                st["attempt"] += 1
                st["user"] = st["base_user"] + "\n\n" + parsed.feedback
        return final

    def commit(self) -> None:
        """Apply the actions once every waiting agent has one."""
        if not self.waiting or any(st["record"] is None for st in self.waiting.values()):
            return
        actions = {aid: st["record"]["action"] for aid, st in self.waiting.items()}
        step = self.env.submit(actions)
        self.last_rewards, self.last_info = step.rewards, step.info
        for aid, r in step.rewards.items():
            self.total_rewards[aid] = self.total_rewards.get(aid, 0.0) + float(r)
        self._collect_eliminations(step.info)
        for aid, st in self.waiting.items():
            rec, res, steer = st["record"], st["record"]["result"], self.steers.get(aid)
            self.logger.log_step(
                game=self.job.game, turn=self.turn, agent_id=aid, system_prompt=st["system"],
                user_prompt=st["user"], completion=res.text, parsed_action=rec["action"],
                parse_retries=st["attempt"], parse_ok=rec["parse_ok"], reward=step.rewards.get(aid, 0.0),
                steering=steer.to_log() if steer else None, info=step.info,
                finish_reason=res.finish_reason, n_tokens=res.n_tokens, monitor=rec["monitor"],
                images=st["images"], attempts=st["attempts"], model_input=res.prompt,
                tokens_in=sum(a["n_prompt_tokens"] for a in st["attempts"]),  # every attempt: retries re-read
                tokens_out=sum(a["n_tokens"] for a in st["attempts"]))
        self.waiting = {}
        self.turn += 1
        if step.done or self.turn >= self.job.max_turns:
            self.finish(truncated=not step.done)

    def _collect_eliminations(self, info: Dict[str, Any]) -> None:
        seen = {e["player_id"] for e in self.eliminations}
        for key in _ELIM_STEP_KEYS:
            val = info.get(key)
            pid = f"player_{val}" if isinstance(val, int) else (str(val) if val is not None else None)
            if pid and pid not in seen:
                seen.add(pid)
                self.eliminations.append({"player_id": pid, "turn": self.turn, "type": key})

    def _roles(self) -> Dict[str, str]:
        roles = self.env.roles() if hasattr(self.env, "roles") else {}
        close_info = self.last_info.get("close_info") or {}
        if not roles and isinstance(close_info, dict):
            for key in ("roles", "player_roles", "role_assignments", "role_map", "assignments"):
                val = close_info.get(key)
                if isinstance(val, dict) and val:
                    return {(f"player_{k}" if isinstance(k, int) else str(k)): str(v) for k, v in val.items()}
        return roles

    def finish(self, truncated: bool = False) -> None:
        close_info = self.last_info.get("close_info") or {}
        if isinstance(close_info, dict):
            seen = {e["player_id"] for e in self.eliminations}
            for key in _ELIM_CLOSE_KEYS:
                if isinstance(close_info.get(key), list) and close_info[key]:
                    for raw in close_info[key]:
                        pid = f"player_{raw}" if isinstance(raw, int) else str(raw)
                        if pid not in seen:  # turn -1: eliminated, but not known when
                            seen.add(pid)
                            self.eliminations.append({"player_id": pid, "turn": -1, "type": key})
                    break
        self.logger.close(summary={
            "total_rewards": self.total_rewards, "final_rewards": self.last_rewards, "final_info": self.last_info,
            "player_roles": self._roles(),
            "teams": self.env.teams() if hasattr(self.env, "teams") else {},
            "eliminations": self.eliminations, "turns": self.turn, "truncated": truncated,
            "steered": {aid: s.to_log() for aid, s in self.steers.items()},
        })
        self.done = True
        self.release()

    def release(self) -> None:
        """Free the game's resources (e.g. a Melting Pot worker process)."""
        try:
            self.env.close()
        except Exception:  # noqa: BLE001 - closing must never take other episodes down
            log(f"warning: closing {self.job.run_id} episode {self.job.episode} failed")


def run_jobs(jobs: List[Job], generator: Generator, reader: Optional[Reader] = None,
             monitor: Optional[Monitor] = None, concurrent: int = 32) -> Dict[str, int]:
    """Play every job, `concurrent` episodes at a time. Returns {"ok": n, "failed": n}."""
    queue, active, counts = deque(jobs), [], {"ok": 0, "failed": 0}

    def fail(run, job, stage):
        counts["failed"] += 1
        log(f"ERROR {job.run_id} episode {job.episode} ({stage}):\n{traceback.format_exc()}")
        if run is not None:
            run.logger.close()
            run.release()

    while queue or active:
        while queue and len(active) < concurrent:
            job = queue.popleft()
            try:
                active.append(EpisodeRun(job))
            except Exception:
                fail(None, job, "setup")
        requests = []
        for run in list(active):
            try:
                mine = run.requests()
                if not mine:
                    raise RuntimeError("game is not done but no agent is pending")
                requests += mine
            except Exception:
                active.remove(run)
                fail(run, run.job, "render")
        if not requests:
            continue
        results = {res.key: res for res in generator.generate(requests)}
        final, broken = [], set()
        for run in active:
            try:
                final += run.deliver(results)
            except Exception:
                broken.add(id(run))
                fail(run, run.job, "parse")
        if monitor is not None and final:
            for rec, rows in zip(final, reader.read([rec["result"] for rec in final])):
                rec["monitor"] = monitor.scores(rows)
        for run in list(active):
            if id(run) in broken:
                active.remove(run)
                continue
            try:
                run.commit()
            except Exception:
                active.remove(run)
                fail(run, run.job, f"turn {run.turn}")
                continue
            if run.done:
                active.remove(run)
                counts["ok"] += 1
                log(f"{run.job.run_id} episode {run.job.episode} done after {run.turn} turns: {run.total_rewards}")
    return counts
