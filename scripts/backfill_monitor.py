"""Monitor scores for episodes played without the monitor (`monitor: null` or run_games --no-monitor).

    sbatch scripts/slurm/backfill_monitor.sbatch config/runs/focus_qwen3.8-27b/gbs_6p
    python scripts/backfill_monitor.py --config-dir <group dir> [--shard i --num-shards n] [--engine JSON]
    python scripts/backfill_monitor.py --config-dir config/runs/focus/gbs_3p --verify 10   # against live scores

The live monitor reads every final turn with the unsteered model, prefilled with the turn's exact prompt and answer
(testbed/mia_engine.py, CaptureEngine). This does the same from the JSONL: `model_input` (the exact templated prompt)
and `completion`, with the turn's images from disk (save_images: true; an episode with an image turn whose images were
not saved cannot be read and is skipped).
  text turns   live: the prompt's token ids + the generated token ids. Here: the prompt encoded as the steer engine
               encoded it (the same ids), the answer re-encoded, plus the stop token when the answer ended on one
               (the generated ids include it): the same tokens whenever re-encoding the answer gives the generated
               ones back; turns where the count differs from the logged n_tokens are counted (`mismatched`).
  image turns  as live: prompt + answer text with the images, the answer's rows kept.
Layers and personas: as the live monitor with `monitor: {layers: auto}` (the store's candidate layers, every persona).

Writes episode_N.monitor.json next to the summary: {"layers", "turns": [per JSONL line: {layer: {trait: score}}],
"agent_stats": {agent: {monitor, phase_layer, phases}}, "mismatched": n, "source"}. scripts/summarize_runs.py takes
the monitor from the summary's agent_stats, else from this file. Episodes that have either are skipped, so it can be
rerun; shards like run_games (--shard / --num-shards, episode-major).
--verify N: N episodes that have live scores are read again and compared turn by turn (nothing is written).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testbed.config import RunConfig, common  # noqa: E402
from testbed.engine import Monitor, TurnResult  # noqa: E402
from testbed.logging_.agent_stats import AgentStats  # noqa: E402
from testbed.logging_.episode_logger import retry_io  # noqa: E402
from testbed.steering import VectorStore  # noqa: E402
from testbed.util import log  # noqa: E402

CHUNK = 32  # turns per read: the captured rows of a chunk are scored before the next is read


def side_path(summary_path: Path) -> Path:
    return summary_path.with_name(summary_path.name[:-len(".summary.json")] + ".monitor.json")


def episodes(configs):
    """[(config, episode, summary path)] of every finished episode, episode-major (as run_games deals them)."""
    out = []
    for cfg in configs:
        dirs = [cfg.run_dir] + ([cfg.fallback_logging_dir / cfg.run_id] if cfg.fallback_logging_dir else [])
        seen = set()
        for d in dirs:
            for spath in sorted(retry_io(lambda d=d: list(d.glob("episode_*.summary.json")))):
                ep = int(spath.name.split("_")[1].split(".")[0])
                if ep not in seen:
                    seen.add(ep)
                    out.append((cfg, ep, spath))
    return sorted(out, key=lambda x: (x[1], x[0].run_id))


def has_live_monitor(summary: dict) -> bool:
    return any(a.get("monitor") for a in (summary.get("agent_stats") or {}).values())


def turn_result(rec: dict, run_dir: Path, encode, stop_id: int):
    """(TurnResult as the capture engine reads it, whether its answer re-encodes to a different token count) of one
    JSONL record; None when the turn cannot be read (no model_input, or images that were not saved)."""
    from PIL import Image

    prompt, text, n_gen = rec.get("model_input") or "", rec.get("completion") or "", rec.get("n_tokens") or 0
    key = (rec["turn"], rec["agent_id"])
    images = rec.get("images") or []
    if isinstance(images, int):  # only a count (save_images off): fine for a text turn, unreadable with images
        if images:
            return None, False
        images = []
    if not n_gen:  # nothing generated: no readout, as live
        return TurnResult(key, text, rec.get("finish_reason", "stop")), False
    if not prompt:
        return None, False
    if images:
        loaded = []
        for rel in images:
            with Image.open(run_dir / rel) as im:
                loaded.append(im.convert("RGB"))
        n_text = len(encode(prompt + text)) - len(encode(prompt))
        attempts = rec.get("attempts") or [{}]
        res = TurnResult(key, text, rec.get("finish_reason", "stop"), [], [0] * n_text, prompt, loaded,
                         attempts[-1].get("n_prompt_tokens") or len(encode(prompt)))
        return res, n_text not in (n_gen, n_gen - 1)
    ids = encode(text)
    if rec.get("finish_reason") == "stop" and len(ids) == n_gen - 1:
        ids = ids + [stop_id]
    return TurnResult(key, text, rec.get("finish_reason", "stop"), encode(prompt), ids, prompt), len(ids) != n_gen


def read_episode(spath: Path, reader, monitor, encode, stop_id):
    """(per-line scores, mismatched turns) of one episode, or None when one of its turns cannot be read."""
    jsonl = spath.with_name(spath.name[:-len(".summary.json")] + ".jsonl")
    recs = [json.loads(line) for line in retry_io(lambda: jsonl.read_text(encoding="utf-8")).splitlines() if line]
    turns = [turn_result(rec, spath.parent, encode, stop_id) for rec in recs]
    if any(t is None for t, _ in turns):
        return None
    scores = []
    for i in range(0, len(turns), CHUNK):
        rows = reader.read([t for t, _ in turns[i:i + CHUNK]])
        scores += [monitor.scores(r) for r in rows]
    return recs, scores, sum(bad for _, bad in turns)


def compare(recs, scores) -> dict:
    """Live vs re-read scores over the turns that have both: max and mean |difference|, and the live scores' spread."""
    import statistics
    diffs, live = [], []
    for rec, new in zip(recs, scores):
        for layer, traits in (rec.get("monitor") or {}).items():
            for t, v in traits.items():
                if t in new.get(layer, {}):
                    diffs.append(abs(new[layer][t] - v))
                    live.append(v)
    if not diffs:
        return {}
    return {"n": len(diffs), "max": max(diffs), "mean": statistics.mean(diffs),
            "live_sd": statistics.pstdev(live) if len(live) > 1 else float("nan")}


def write_side(spath: Path, recs, scores, layers, mismatched) -> None:
    stats = AgentStats()
    for rec, s in zip(recs, scores):
        stats.add_record({**rec, "monitor": s})
    per_agent = {aid: {k: a[k] for k in ("monitor", "phase_layer", "phases")} for aid, a in stats.to_json().items()}
    side = side_path(spath)
    tmp = side.with_name(f"{side.name}.{os.getpid()}.tmp")
    data = {"layers": layers, "turns": scores, "agent_stats": per_agent, "mismatched": mismatched,
            "source": "scripts/backfill_monitor.py"}
    retry_io(lambda: tmp.write_text(json.dumps(data), encoding="utf-8"))
    retry_io(os.replace, tmp, side)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", required=True, help="one game group's configs (they share the model)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--engine", type=json.loads, help="JSON merged into the configs' engine settings (the monitor's "
                   "GPUs, memory, buffers), as run_games --engine")
    p.add_argument("--verify", type=int, metavar="N", help="re-read N episodes that have live scores and compare")
    args = p.parse_args(argv)

    configs = [RunConfig.load(x) for x in sorted(str(x) for x in Path(args.config_dir).rglob("*.yaml"))]
    model, engine_cfg = common(configs, "model"), common(configs, "engine")
    store = VectorStore(common(configs, "vectors_dir"), cache_dir=Path(configs[0].logging_dir) / "_vectors" / model,
                        traits_dir=common(configs, "traits_dir"))
    layers = store.candidate_layers()
    monitor = Monitor(store, layers, None)
    todo = episodes(configs)[args.shard::args.num_shards]
    log(f"{len(configs)} run(s), {len(todo)} finished episode(s) in shard {args.shard}/{args.num_shards} of "
        f"{Path(args.config_dir).as_posix().rstrip('/')}; monitor layers {layers}")

    engine_cfg = {**engine_cfg, "max_images": engine_cfg.get("max_images", 2)
                  if any(c.observation != "text" for c in configs) else 0}
    for k, v in (args.engine or {}).items():
        engine_cfg = {**engine_cfg, k: {**(engine_cfg.get(k) or {}), **v} if isinstance(v, dict) else v}
    engine_cfg = {**engine_cfg, "steer": {"off": True}}

    from persvecfast.models import MODELS
    from transformers import AutoTokenizer
    from testbed.mia_engine import open_engines
    tok = AutoTokenizer.from_pretrained(MODELS[model].hf_repo)
    encode = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731 - as the steer engine encodes

    counts = {"written": 0, "skipped": 0, "unreadable": 0, "mismatched turns": 0}
    checks = []
    with open_engines(model, engine_cfg, layers) as (_, reader):
        for cfg, ep, spath in todo:
            summary = json.loads(retry_io(lambda: spath.read_text(encoding="utf-8")))
            live = has_live_monitor(summary)
            if (live or side_path(spath).exists()) if not args.verify else not live:
                counts["skipped"] += 1
                continue
            got = read_episode(spath, reader, monitor, encode, tok.eos_token_id)
            if got is None:
                counts["unreadable"] += 1
                log(f"{cfg.run_id} episode {ep}: a turn cannot be read (images not saved, or no model_input)")
                continue
            recs, scores, mismatched = got
            counts["mismatched turns"] += mismatched
            if args.verify:
                checks.append(compare(recs, scores))
                c = checks[-1]
                log(f"{cfg.run_id} episode {ep}: {c.get('n', 0)} scores, |re-read - live| max {c.get('max', 0):.4f}, "
                    f"mean {c.get('mean', 0):.4f} (live SD {c.get('live_sd', float('nan')):.3f}); "
                    f"{mismatched} turn(s) re-encoded to another length")
                if len(checks) >= args.verify:
                    break
                continue
            write_side(spath, recs, scores, layers, mismatched)
            counts["written"] += 1
            if counts["written"] % 50 == 0:
                log(f"  {counts['written']} episodes written")
    log("done: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    if checks:
        checks = [c for c in checks if c]
        log(f"verify: {len(checks)} episode(s); max |re-read - live| {max(c['max'] for c in checks):.4f}, mean "
            f"{sum(c['mean'] for c in checks) / len(checks):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
