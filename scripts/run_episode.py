"""CLI: run episode(s) from a YAML config."""
from __future__ import annotations

import argparse
import os
import sys

import yaml

# allow running as a script: ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testbed.config import RunConfig, build_policy, build_steering  # noqa: E402
from testbed.logging_.episode_logger import EpisodeLogger  # noqa: E402
from testbed.orchestrator import Orchestrator  # noqa: E402
from testbed.registry import build_game  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = RunConfig.from_dict(raw)

    steering = build_steering(cfg.steering)
    policy = build_policy(cfg.model, steering=steering)

    for ep in range(cfg.episodes):
        env, renderer, parser_ = build_game(
            family=cfg.game_family, game_id=cfg.game_id,
            num_players=cfg.num_players or 3, env_kwargs=cfg.env_kwargs)
        logger = EpisodeLogger(run_dir=cfg.logging_dir, run_id=cfg.run_id, episode=ep)
        orch = Orchestrator(
            env=env, renderer=renderer, parser=parser_, policy=policy,
            steering=steering, logger=logger, game=cfg.game_id,
            max_parse_retries=cfg.max_parse_retries)
        final = orch.run_episode()
        print(f"Episode {ep} final rewards: {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
