"""CLI: run episode(s) from a YAML config."""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import zlib

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


def _completed_episodes(logging_dir: str, run_id: str) -> set[int]:
    """Episode indices that already have a summary.json — i.e. fully logged.

    Used to resume a run across separate job submissions (e.g. a SLURM job
    that hit its walltime) without re-running or overwriting episodes that
    already completed successfully.
    """
    run_dir = os.path.join(logging_dir, run_id)
    done = set()
    for path in glob.glob(os.path.join(run_dir, "episode_*.summary.json")):
        m = re.search(r"episode_(\d+)\.summary\.json$", os.path.basename(path))
        if m:
            done.add(int(m.group(1)))
    return done


def _episode_env_kwargs(cfg: RunConfig, ep: int) -> dict:
    """Per-episode env kwargs, with a per-episode GBS seed injected.

    GBSAdapter defaults to seed=0 when not given one, so without this every
    episode (and every run_id/task sharing a player count) would draw the
    identical hidden target. Deriving the seed from (run_id, episode) instead
    gives each episode its own target, and different tasks (e.g. different
    reasoning-sweep modes) their own independent sequence of targets, while
    staying deterministic across resumed/rerun chunks of the same episode.
    A config that already pins its own `seed` or `target` is left alone.
    """
    kwargs = dict(cfg.env_kwargs)
    if cfg.game_id == "gbs" and "seed" not in kwargs and "target" not in kwargs:
        kwargs["seed"] = zlib.crc32(f"{cfg.run_id}:{ep}".encode()) & 0xFFFFFFFF
    return kwargs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = RunConfig.from_dict(raw)

    completed = _completed_episodes(cfg.logging_dir, cfg.run_id)
    if completed:
        print(f"Resuming {cfg.run_id}: {len(completed)}/{cfg.episodes} "
              f"episode(s) already complete, skipping those.")

    wandb_run = _init_wandb(cfg, raw)
    steering = build_steering(cfg.steering)
    policy = build_policy(cfg.model, steering=steering)

    try:
        for ep in range(cfg.episodes):
            if ep in completed:
                continue
            env, renderer, parser_ = build_game(
                family=cfg.game_family, game_id=cfg.game_id,
                num_players=cfg.num_players or 3,
                env_kwargs=_episode_env_kwargs(cfg, ep))
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
