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
            "probing": cfg.probing,
            "env_kwargs": cfg.env_kwargs,
        },
    )


def _completed_episodes(logging_dir: str, run_id: str) -> set[int]:
    """Episode indices that already have a summary.json (used to resume runs)."""
    run_dir = os.path.join(logging_dir, run_id)
    done = set()
    for path in glob.glob(os.path.join(run_dir, "episode_*.summary.json")):
        m = re.search(r"episode_(\d+)\.summary\.json$", os.path.basename(path))
        if m:
            done.add(int(m.group(1)))
    return done


def _episode_env_kwargs(cfg: RunConfig, ep: int) -> dict:
    """Per-episode env kwargs with a deterministic per-episode seed injected."""
    kwargs = dict(cfg.env_kwargs)
    if cfg.game_family == "symbolic":
        if "seed" not in kwargs:
            kwargs["seed"] = zlib.crc32(f"{cfg.run_id}:{ep}".encode()) & 0xFFFFFFFF
    elif cfg.game_family == "textarena":
        if "seed" not in kwargs:
            # Intentionally run-ID-independent: all runs see the same topic,
            # side-assignment, and role-assignment for each episode index.
            kwargs["seed"] = zlib.crc32(f"ta_episode:{ep}".encode()) & 0xFFFFFFFF
    return kwargs


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episode count from config.")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()

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
    policy = build_policy(cfg.model, steering=steering, probing_cfg=cfg.probing)

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
