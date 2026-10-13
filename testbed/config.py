"""Run config (one YAML per run) and job building.

    run_id: werewolf_8p__team-evil__evil_x2.5
    game: {family: symbolic, id: werewolf, players: 8, env_kwargs: {}}
    episodes: 50
    model: qwen3.5-9b               # PersVecFast model key
    vectors_dir: ...                # default ${PVF_OUT}/persona/<model>/bf16
    traits_dir: ...                 # default <PersVecFast>/data/persona_vectors (prompt personas)
    max_tokens: 1024                # per turn
    observation: text               # text | image | both: how the game state is shown
    image_size: 448                 # longer side of image observations (pixels)
    save_images: false              # write every turn's images next to the logs
    max_parse_retries: 3
    steering:                       # a group or a list of groups; omit for no steering
      team: evil                    #   pool: team / roles / agents (default: everyone)
      select: all                   #   all | one | count | fraction
      trait: evil                   #   or traits: {evil: 1.0, sarcastic: 0.5}
      coefficient: 2.5
      layer: auto                   #   int, or the trait's best PersVecFast layer
      via: vector                   #   vector | prompt (the trait's instruction) | both
    monitor: {layers: auto, traits: null}   # omit to disable; null traits = all
    logging_dir: logs/
    engine:                         # per job; every config of one job must agree
      concurrent_episodes: 32
      max_model_len: 32768
      steer:   {gpus: "0", memory: 0.85}
      monitor: {gpus: "1", memory: 0.75}
      sampling: {}                  # overrides of the model's vendor sampling settings
      max_images: 2                 # images per prompt the engines accept (0 when every run is text)

Episode seeds depend only on (game, players, episode), never on the run, so every
steering condition plays the same deals, roles and targets: conditions are paired.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from testbed.logging_.episode_logger import EpisodeLogger
from testbed.registry import build_game
from testbed.runner import Job
from testbed.steering import SteeringPlan, VectorStore
from testbed.util import stable_seed, write_json
from testbed.visual import OBSERVATION_MODES

DEFAULT_PVF_OUT = "/scratch/inf0/user/nzukowsk/PersVecFast/out"


@dataclass
class RunConfig:
    run_id: str
    game_family: str
    game_id: str
    players: int
    episodes: int
    model: str
    vectors_dir: str
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    traits_dir: Optional[str] = None
    max_tokens: int = 1024
    max_parse_retries: int = 3
    max_turns: int = 1000
    observation: str = "text"
    image_size: int = 448
    save_images: bool = False
    steering: Any = None
    monitor: Optional[Dict[str, Any]] = None
    logging_dir: str = "logs/"
    engine: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        game = d["game"]
        model = d["model"]
        vectors = d.get("vectors_dir") or f"{os.environ.get('PVF_OUT', DEFAULT_PVF_OUT)}/persona/{model}/bf16"
        return cls(run_id=d["run_id"], game_family=game["family"], game_id=game["id"],
                   players=int(game["players"]), episodes=int(d.get("episodes", 1)), model=model,
                   vectors_dir=os.path.expandvars(vectors), env_kwargs=dict(game.get("env_kwargs") or {}),
                   traits_dir=os.path.expandvars(d["traits_dir"]) if d.get("traits_dir") else None,
                   max_tokens=int(d.get("max_tokens", 1024)),
                   max_parse_retries=int(d.get("max_parse_retries", 3)),
                   max_turns=int(d.get("max_turns", 1000)),
                   observation=_observation(d.get("observation", "text")),
                   image_size=int(d.get("image_size", 448)), save_images=bool(d.get("save_images", False)),
                   steering=d.get("steering"), monitor=d.get("monitor"),
                   logging_dir=d.get("logging_dir", "logs/"), engine=dict(d.get("engine") or {}), raw=d)

    @classmethod
    def load(cls, path) -> "RunConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @property
    def run_dir(self) -> Path:
        return Path(self.logging_dir) / self.run_id

    def episode_seed(self, ep: int) -> int:
        return stable_seed("episode", self.game_id, self.players, ep)

    def completed_episodes(self) -> set:
        return {int(p.name.split("_")[1].split(".")[0]) for p in self.run_dir.glob("episode_*.summary.json")}


def _observation(mode: str) -> str:
    if mode not in OBSERVATION_MODES:
        raise ValueError(f"observation must be one of {OBSERVATION_MODES}, got {mode!r}")
    return mode


def common(configs: List[RunConfig], what: str):
    """The value of `what` shared by every config of one job (they share one engine pair)."""
    values = {repr(getattr(c, what)) for c in configs}
    if len(values) > 1:
        raise ValueError(f"configs of one job must share `{what}`, got {sorted(values)}")
    return getattr(configs[0], what)


def monitor_layers(monitor_cfg, store: VectorStore) -> Optional[List[int]]:
    if not monitor_cfg:
        return None
    layers = monitor_cfg.get("layers", "auto")
    return store.candidate_layers() if layers == "auto" else [int(l) for l in layers]


def build_jobs(configs: List[RunConfig], store: VectorStore, shard: int = 0, num_shards: int = 1,
               episodes: Optional[int] = None) -> List[Job]:
    """Jobs for one shard of every run's episodes, minus the finished ones.

    Episodes are ordered episode-major (episode 0 of every run, then episode 1, ...) and cut
    into contiguous shards, so each shard covers every condition; sharding happens before
    finished episodes are dropped, so an episode always belongs to the same shard."""
    plans, order = [], []
    for cfg in configs:
        plan = SteeringPlan.from_config(cfg.steering, store)
        plans.append(plan)
        write_json(cfg.run_dir / "run.json", {**cfg.raw, "vectors_dir": cfg.vectors_dir,
                                              "steering_resolved": [s.to_log() for s in plan.steers]})
    counts = [episodes if episodes is not None else cfg.episodes for cfg in configs]
    for ep in range(max(counts, default=0)):
        order += [(i, ep) for i in range(len(configs)) if ep < counts[i]]
    mine = order[len(order) * shard // num_shards:len(order) * (shard + 1) // num_shards]
    done = [cfg.completed_episodes() for cfg in configs]
    return [_job(configs[i], i, ep, plans[i]) for i, ep in mine if ep not in done[i]]


def _job(cfg: RunConfig, index: int, ep: int, plan: SteeringPlan) -> Job:
    seed = cfg.episode_seed(ep)

    def make_env():
        env, renderer, parser = build_game(family=cfg.game_family, game_id=cfg.game_id,
                                           num_players=cfg.players, env_kwargs={"seed": seed, **cfg.env_kwargs})
        env.reset()
        return env, renderer, parser

    return Job(run_index=index, run_id=cfg.run_id, game=cfg.game_id, episode=ep, make_env=make_env,
               assign=lambda env: plan.assign(env, stable_seed("steer", seed)),
               make_logger=lambda: EpisodeLogger(run_dir=cfg.logging_dir, run_id=cfg.run_id, episode=ep,
                                                 save_images=cfg.save_images),
               max_parse_retries=cfg.max_parse_retries, max_tokens=cfg.max_tokens, max_turns=cfg.max_turns,
               observation=cfg.observation, image_size=cfg.image_size)
