"""CLI: run episode(s) from a YAML config."""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testbed.config import RunConfig, build_policy, build_steering  # noqa: E402
from testbed.logging_.episode_logger import EpisodeLogger  # noqa: E402
from testbed.orchestrator import Orchestrator  # noqa: E402
from testbed.registry import build_game  # noqa: E402


def _init_wandb(cfg: RunConfig, raw: dict):
    """Return a wandb run if wandb is configured, else None."""
    wcfg = raw.get("wandb", {})
    if not wcfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping. pip install wandb to enable.")
        return None
    return wandb.init(
        project=wcfg.get("project", "ma-steering"),
        name=wcfg.get("name", cfg.run_id),
        tags=wcfg.get("tags", []),
        config={
            "run_id": cfg.run_id,
            "game_family": cfg.game_family,
            "game_id": cfg.game_id,
            "episodes": cfg.episodes,
            "num_players": cfg.num_players,
            "model": cfg.model,
            "steering": cfg.steering,
            "env_kwargs": cfg.env_kwargs,
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = RunConfig.from_dict(raw)

    wandb_run = _init_wandb(cfg, raw)
    steering = build_steering(cfg.steering)
    policy = build_policy(cfg.model, steering=steering)

    try:
        for ep in range(cfg.episodes):
            env, renderer, parser_ = build_game(
                family=cfg.game_family, game_id=cfg.game_id,
                num_players=cfg.num_players or 3, env_kwargs=cfg.env_kwargs)
            logger = EpisodeLogger(
                run_dir=cfg.logging_dir, run_id=cfg.run_id,
                episode=ep, wandb_run=wandb_run)
            orch = Orchestrator(
                env=env, renderer=renderer, parser=parser_, policy=policy,
                steering=steering, logger=logger, game=cfg.game_id,
                max_parse_retries=cfg.max_parse_retries)
            final = orch.run_episode()
            print(f"Episode {ep} final rewards: {final}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
