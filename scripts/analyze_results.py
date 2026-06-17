"""Aggregate episode JSONL logs into a summary CSV for cross-run comparison.

Reads every episode_N.summary.json under --logs-dir, infers game / layer /
steering from the run_id, and writes a flat CSV with one row per episode.

Usage
-----
python scripts/analyze_results.py --logs-dir logs/ --output results_summary.csv
python scripts/analyze_results.py --logs-dir logs/ --print-table
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


# ── run_id parsing ────────────────────────────────────────────────────────────

def _parse_run_id(run_id: str) -> dict:
    """Extract (game, steering, scope, layer) from a run_id string."""
    info = {"game": "beauty_contest", "steering": "noop",
            "scope": "all", "layer": None}

    if run_id.startswith("gbs_"):
        info["game"] = "gbs"

    if "activation" in run_id:
        info["steering"] = "activation"
    elif "prompt" in run_id:
        info["steering"] = "prompt"
    else:
        info["steering"] = "noop"

    if "_one" in run_id:
        info["scope"] = "one"
    else:
        info["scope"] = "all"

    m = re.search(r"_l(\d+)$", run_id)
    if m:
        info["layer"] = int(m.group(1))

    return info


# ── per-episode metrics ───────────────────────────────────────────────────────

def _episode_metrics(summary: dict, jsonl_path: Path) -> dict:
    """Derive per-episode metrics from summary + JSONL step records."""
    metrics: dict = {}

    final_rewards = summary.get("final_rewards", {})
    if final_rewards:
        rewards = list(final_rewards.values())
        metrics["mean_reward"] = sum(rewards) / len(rewards)
        metrics["max_reward"] = max(rewards)
        # player_0 reward — the steered agent in "one" configs
        metrics["player_0_reward"] = final_rewards.get("player_0", 0.0)

    metrics["gbs_converged"] = summary.get("gbs_converged", None)
    metrics["gbs_converged_round"] = summary.get("gbs_converged_round", None)

    # aggregate step-level info from JSONL
    if not jsonl_path.exists():
        return metrics

    guesses, errors, abs_errors, contributions = [], [], [], []
    parse_retries_total = 0
    n_steps = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_steps += 1
            parse_retries_total += rec.get("parse_retries", 0)
            info = rec.get("info", {})
            action = rec.get("parsed_action")
            game = rec.get("game", "")

            if game == "beauty_contest" and action is not None:
                try:
                    guesses.append(float(action))
                except (TypeError, ValueError):
                    pass
                if "target" in info:
                    try:
                        errors.append(float(action) - info["target"])
                    except (TypeError, ValueError):
                        pass

            elif game == "gbs" and action is not None:
                try:
                    contributions.append(float(action))
                except (TypeError, ValueError):
                    pass
                if "error" in info:
                    abs_errors.append(abs(info["error"]))

    metrics["n_steps"] = n_steps
    metrics["parse_retries_total"] = parse_retries_total

    if guesses:
        metrics["mean_guess"] = sum(guesses) / len(guesses)
        metrics["min_guess"] = min(guesses)
        metrics["max_guess"] = max(guesses)
    if errors:
        metrics["mean_guess_error"] = sum(abs(e) for e in errors) / len(errors)
    if contributions:
        metrics["mean_contribution"] = sum(contributions) / len(contributions)
    if abs_errors:
        metrics["mean_abs_gbs_error"] = sum(abs_errors) / len(abs_errors)
        metrics["final_abs_gbs_error"] = abs_errors[-1] if abs_errors else None

    return metrics


# ── main ─────────────────────────────────────────────────────────────────────

def collect_rows(logs_dir: str) -> list[dict]:
    rows = []
    root = Path(logs_dir)
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        parsed = _parse_run_id(run_id)

        for summary_path in sorted(run_dir.glob("episode_*.summary.json")):
            m = re.search(r"episode_(\d+)\.summary\.json$", summary_path.name)
            ep = int(m.group(1)) if m else -1
            jsonl_path = summary_path.with_suffix("").with_suffix(".jsonl")

            try:
                with open(summary_path, encoding="utf-8") as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            ep_metrics = _episode_metrics(summary, jsonl_path)

            row = {
                "run_id": run_id,
                "game": parsed["game"],
                "steering": parsed["steering"],
                "scope": parsed["scope"],
                "layer": parsed["layer"],
                "episode": ep,
                **ep_metrics,
            }
            rows.append(row)

    return rows


def _fmt(val, fmt) -> str:
    """Format a value, showing '-' for nan/None."""
    if val is None:
        return "-"
    try:
        if val != val:  # nan check
            return "-"
        return format(val, fmt)
    except (TypeError, ValueError):
        return "-"


def print_table(rows: list[dict]) -> None:
    """Print a grouped summary: mean metrics per (game, steering, scope, layer)."""
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for r in rows:
        key = (r["game"], r["steering"], r["scope"], r["layer"])
        groups[key].append(r)

    # sort with None layer last (baselines)
    def sort_key(k):
        game, steering, scope, layer = k
        return (game, steering, scope, layer if layer is not None else 9999)

    print(f"\n{'game':<18} {'steering':<12} {'scope':<6} {'layer':<7} "
          f"{'n_ep':<6} {'mean_reward':<13} {'mean_guess':<12} "
          f"{'gbs_conv%':<10} {'gbs_conv_rd':<12} {'mean_abs_err'}")
    print("-" * 110)

    for key in sorted(groups, key=sort_key):
        game, steering, scope, layer = key
        grp = groups[key]
        n = len(grp)

        def avg(field):
            vals = [r[field] for r in grp if r.get(field) is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        def pct(field):
            vals = [r[field] for r in grp if r.get(field) is not None]
            return 100 * sum(bool(v) for v in vals) / len(vals) if vals else float("nan")

        layer_str = str(layer) if layer is not None else "-"
        print(f"{game:<18} {steering:<12} {scope:<6} {layer_str:<7} "
              f"{n:<6} {_fmt(avg('mean_reward'), '.3f'):<13} "
              f"{_fmt(avg('mean_guess'), '.1f'):<12} "
              f"{_fmt(pct('gbs_converged'), '.1f'):<10} "
              f"{_fmt(avg('gbs_converged_round'), '.1f'):<12} "
              f"{_fmt(avg('mean_abs_gbs_error'), '.2f')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="logs/",
                    help="Root logs directory (default: logs/)")
    ap.add_argument("--output", default="results_summary.csv",
                    help="Output CSV path (default: results_summary.csv)")
    ap.add_argument("--print-table", action="store_true",
                    help="Also print a grouped summary table to stdout")
    args = ap.parse_args()

    rows = collect_rows(args.logs_dir)
    if not rows:
        print(f"No episode summaries found under {args.logs_dir}")
        sys.exit(1)

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = (["run_id", "game", "steering", "scope", "layer", "episode"] +
                  sorted(all_keys - {"run_id", "game", "steering",
                                     "scope", "layer", "episode"}))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows → {args.output}")

    if args.print_table:
        print_table(rows)


if __name__ == "__main__":
    main()
