"""Play games with persona-vector steering and live monitoring on vLLM-Hook.

    python scripts/run_games.py --config config/runs/main/werewolf_8p__none.yaml
    python scripts/run_games.py --config-dir config/runs/main --shard 3 --num-shards 20

All configs given to one process share one engine pair (same model, monitor and engine
settings) and play concurrently: every scheduler step batches the pending turns of up to
`concurrent_episodes` episodes, whatever run they belong to. Rerunning resumes: episodes
with a summary file are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testbed.config import RunConfig, build_jobs, common, monitor_layers  # noqa: E402
from testbed.engine import Monitor  # noqa: E402
from testbed.runner import run_jobs  # noqa: E402
from testbed.steering import VectorStore  # noqa: E402
from testbed.util import log  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # OPENROUTER_API_KEY for TextArena juries


def main(argv=None, open_engines=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="*", default=[])
    p.add_argument("--config-dir", help="every *.yaml below this directory")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--episodes", type=int, help="override every config's episode count (smoke tests)")
    p.add_argument("--concurrent", type=int, help="override engine.concurrent_episodes")
    args = p.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv()  # OPENROUTER_API_KEY for TextArena juries

    paths = list(args.config) + (sorted(str(x) for x in Path(args.config_dir).rglob("*.yaml")) if args.config_dir else [])
    if not paths:
        p.error("give --config and/or --config-dir")
    configs = [RunConfig.load(x) for x in paths]
    if len({c.run_id for c in configs}) != len(configs):
        p.error("duplicate run_id among the configs")
    model, engine_cfg = common(configs, "model"), common(configs, "engine")
    monitor_cfg, vectors_dir = common(configs, "monitor"), common(configs, "vectors_dir")

    store = VectorStore(vectors_dir, cache_dir=Path(configs[0].logging_dir) / "_vectors" / model,
                        traits_dir=common(configs, "traits_dir"))
    jobs = build_jobs(configs, store, args.shard, args.num_shards, args.episodes)
    log(f"{len(configs)} run(s), {len(jobs)} episode(s) to play in shard {args.shard}/{args.num_shards}"
        + (f" of {Path(args.config_dir).as_posix().rstrip('/')}" if args.config_dir else ""))  # read by progress.py
    if not jobs:
        return 0

    layers = monitor_layers(monitor_cfg, store)
    monitor = Monitor(store, layers, monitor_cfg.get("traits")) if layers is not None else None
    if open_engines is None:
        from testbed.mia_engine import open_engines
    concurrent = args.concurrent or engine_cfg.get("concurrent_episodes", 32)
    visual = any(c.observation != "text" for c in configs)
    engine_cfg = {**engine_cfg, "max_images": engine_cfg.get("max_images", 2) if visual else 0}
    with open_engines(model, engine_cfg, layers) as (generator, reader):
        counts = run_jobs(jobs, generator, reader, monitor, concurrent=concurrent)
    log(f"done: {counts['ok']} episode(s) ok, {counts['failed']} failed")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
