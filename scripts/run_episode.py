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


def _init_wandb(cfg: RunConfig, raw: dict, shard: int, num_shards: int):
    """Return a wandb run if wandb is configured, else None."""
    wcfg = raw.get("wandb", {})
    if not wcfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping. pip install wandb to enable.")
        return None
    name = wcfg.get("name", cfg.run_id)
    if num_shards > 1:
        # distinguish concurrent shards of the same run_id in the wandb UI
        name = f"{name}-shard{shard}"
    return wandb.init(
        project=wcfg.get("project", "ma-steering"),
        name=name,
        tags=wcfg.get("tags", []),
        dir="wandb_logs",
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
    if cfg.game_id in ("gbs", "gbs_exact_replication") and "seed" not in kwargs and "target" not in kwargs:
        # seed_base lets multiple runs share the same random scenario sequence so
        # that per-episode results are directly comparable across conditions.
        seed_base = kwargs.pop("seed_base", cfg.run_id)
        kwargs["seed"] = zlib.crc32(f"{seed_base}:{ep}".encode()) & 0xFFFFFFFF
    else:
        kwargs.pop("seed_base", None)   # harmless cleanup if seed/target was explicit
    return kwargs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard", type=int, default=0,
                        help="This process's shard index (0-based). "
                             "Combined with --num-shards to split one "
                             "run_id's episodes across concurrent processes.")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of concurrent shards for this "
                             "run_id. Episode ep is handled by shard "
                             "ep %% num_shards. Default 1 = no sharding.")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episode count from config. "
                             "Useful for single-episode debugging (--episodes 1).")
    args = parser.parse_args(argv)


    if not (0 <= args.shard < args.num_shards):
        parser.error(f"--shard must be in [0, {args.num_shards}), got {args.shard}")

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = RunConfig.from_dict(raw)
    if args.episodes is not None:
        cfg.episodes = args.episodes

    my_episodes = [ep for ep in range(cfg.episodes) if ep % args.num_shards == args.shard]
    completed = _completed_episodes(cfg.logging_dir, cfg.run_id)
    remaining = [ep for ep in my_episodes if ep not in completed]
    if args.num_shards > 1:
        print(f"Shard {args.shard}/{args.num_shards} of {cfg.run_id}: "
              f"{len(my_episodes)} assigned, {len(remaining)} remaining.")
    elif completed:
        print(f"Resuming {cfg.run_id}: {len(completed)}/{cfg.episodes} "
              f"episode(s) already complete, skipping those.")

    wandb_run = _init_wandb(cfg, raw, args.shard, args.num_shards)
    steering = build_steering(cfg.steering)
    policy = build_policy(cfg.model, steering=steering)

    try:
        for ep in remaining:
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
