"""CLI: run episode(s) from a YAML config."""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import zlib

import yaml
from dotenv import load_dotenv

load_dotenv()

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


# Symbolic games whose adapter does not accept a `seed` kwarg.
_NO_SEED_GAMES = frozenset({"beauty_contest"})


def _episode_env_kwargs(cfg: RunConfig, ep: int) -> dict:
    """Per-episode env kwargs with a deterministic per-episode seed injected.

    Without this, every episode runs with seed=0 and produces structurally
    identical games (same role draws, same starting positions, etc.).
    Deriving the seed from (run_id, episode) gives each episode its own
    randomised setup while staying fully reproducible across resumed runs.

    GBS keeps its special seed_base logic so that different steering conditions
    share the same scenario sequence and results are directly comparable.
    Games listed in _NO_SEED_GAMES have no seed parameter and are left alone.
    TextArena games are passed through unchanged.
    A config that already pins its own seed/target is never overridden.
    """
    kwargs = dict(cfg.env_kwargs)

    if cfg.game_id in ("gbs", "gbs_exact_replication"):
        if "seed" not in kwargs and "target" not in kwargs:
            seed_base = kwargs.pop("seed_base", cfg.run_id)
            kwargs["seed"] = zlib.crc32(f"{seed_base}:{ep}".encode()) & 0xFFFFFFFF
        else:
            kwargs.pop("seed_base", None)
    elif cfg.game_family == "symbolic" and cfg.game_id not in _NO_SEED_GAMES:
        kwargs.pop("seed_base", None)
        if "seed" not in kwargs:
            kwargs["seed"] = zlib.crc32(f"{cfg.run_id}:{ep}".encode()) & 0xFFFFFFFF
    else:
        kwargs.pop("seed_base", None)

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

    n_ok = 0
    n_err = 0
    try:
        for ep in remaining:
            try:
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
                n_ok += 1
            except Exception as exc:
                n_err += 1
                print(f"ERROR episode {ep}: {type(exc).__name__}: {exc}", flush=True)
                import traceback
                traceback.print_exc()
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    if n_err:
        print(f"Completed {n_ok} episodes OK, {n_err} failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
