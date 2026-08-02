"""Qualitative persona probe analysis.

Reads episode JSONL logs from one or two run directories (steered + control)
and produces a qualitative report showing which persona traits each agent
exhibits, how the steered agent differs from the control, and whether the
steering appears to affect other agents.

Usage
─────
# Single run (steered only)
python scripts/textarena/analyze_persona_probes.py \\
    --steered  logs/mafia/mafia_activation_wolf_opportunistic_8p

# Steered vs. control comparison
python scripts/textarena/analyze_persona_probes.py \\
    --steered  logs/mafia/mafia_activation_wolf_opportunistic_8p \\
    --control  logs/mafia/mafia_noop_8p \\
    --top-k 5 \\
    --out reports/mafia_probe_analysis.txt

Output is printed to stdout and optionally saved to --out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: str) -> List[dict]:
    """Load all JSONL step records from a run directory."""
    records = []
    pattern = os.path.join(run_dir, "**", "episode_*.jsonl")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("persona_probe"):
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
    return records


def load_summaries(run_dir: str) -> List[dict]:
    summaries = []
    for path in sorted(glob.glob(os.path.join(run_dir, "**", "episode_*.summary.json"),
                                  recursive=True)):
        try:
            with open(path, encoding="utf-8") as f:
                summaries.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_probes(
    records: List[dict],
) -> Dict[str, Dict[str, float]]:
    """Compute per-agent mean trait projections across all episodes and turns.

    Returns {agent_id: {trait: mean_projection}}.
    """
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, int] = defaultdict(int)

    for rec in records:
        aid = rec["agent_id"]
        probe = rec.get("persona_probe") or {}
        # probe is {"mean": {trait: float}, "chunks": [...]} — use mean for aggregation
        scores_dict = probe.get("mean", probe) if isinstance(probe, dict) else {}
        # skip records where "mean" key is missing (old flat format falls back gracefully)
        if "chunks" in scores_dict:
            scores_dict = {}  # malformed; skip
        for trait, score in scores_dict.items():
            sums[aid][trait] += score
        counts[aid] += 1

    return {
        aid: {trait: total / counts[aid] for trait, total in traits.items()}
        for aid, traits in sums.items()
    }


def top_k(scores: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def steered_agents_from_summaries(summaries: List[dict]) -> Dict[str, int]:
    """Count how often each agent_id was the steered (non-zero) agent."""
    counts: Dict[str, int] = defaultdict(int)
    for s in summaries:
        steered = s.get("steered", 0)
        if steered == 0:
            continue
        roles = s.get("player_roles", {})
        for aid, role in roles.items():
            counts[aid] += 1
    return dict(counts)


# ── Formatting ────────────────────────────────────────────────────────────────

_BAR_WIDTH = 20


def _bar(score: float, max_score: float = 0.5) -> str:
    filled = max(0, min(_BAR_WIDTH, int(score / max_score * _BAR_WIDTH)))
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _fmt_trait(trait: str, score: float, delta: Optional[float] = None,
               highlight: bool = False) -> str:
    bar = _bar(score)
    delta_str = ""
    if delta is not None:
        sign = "+" if delta >= 0 else ""
        marker = " ★" if abs(delta) >= 0.05 else ""
        delta_str = f"  Δ={sign}{delta:.3f}{marker}"
    flag = " ◀ STEERED" if highlight else ""
    return f"    {trait:<35} {score:+.3f}  {bar}{delta_str}{flag}"


def _section(title: str) -> str:
    return f"\n── {title} {'─' * max(0, 68 - len(title))}\n"


# ── Report ────────────────────────────────────────────────────────────────────

def report(
    steered_records: List[dict],
    control_records: Optional[List[dict]],
    steered_summaries: List[dict],
    top_k_n: int,
    out_path: Optional[str],
) -> None:
    lines: List[str] = []
    add = lines.append

    steered_agg = aggregate_probes(steered_records)
    control_agg = aggregate_probes(control_records) if control_records else {}

    n_eps_steered = len({r["episode"] for r in steered_records})
    n_eps_control = len({r["episode"] for r in control_records}) if control_records else 0

    # ── Header ────────────────────────────────────────────────────────────
    add("=" * 72)
    add("  PERSONA PROBE ANALYSIS")
    add("=" * 72)
    add(f"  Steered run  : {n_eps_steered} episode(s), "
        f"{len(steered_records)} logged steps across "
        f"{len(steered_agg)} agent(s)")
    if control_agg:
        add(f"  Control run  : {n_eps_control} episode(s), "
            f"{len(control_records)} logged steps across "
            f"{len(control_agg)} agent(s)")
    add(f"  Top-K traits : {top_k_n}")
    add("")

    # Identify who was steered (appears in steered summaries with steered>0)
    steered_counts = steered_agents_from_summaries(steered_summaries)
    most_steered = max(steered_counts, key=steered_counts.get) if steered_counts else None

    # ── Per-agent breakdown (steered run) ─────────────────────────────────
    add(_section("Per-agent trait projections (steered run)"))
    agents = sorted(steered_agg.keys(),
                    key=lambda a: int("".join(filter(str.isdigit, a)) or "0"))
    for aid in agents:
        scores = steered_agg[aid]
        is_steered_agent = (aid == most_steered)
        label = "  ◀ STEERED AGENT" if is_steered_agent else ""
        add(f"  {aid}{label}")
        top = top_k(scores, top_k_n)
        ctrl_scores = control_agg.get(aid, {})
        for trait, score in top:
            delta = score - ctrl_scores.get(trait, score) if ctrl_scores else None
            add(_fmt_trait(trait, score, delta, highlight=False))
        add("")

    # ── Control run (if provided) ──────────────────────────────────────────
    if control_agg:
        add(_section("Per-agent trait projections (control / noop run)"))
        for aid in agents:
            scores = control_agg.get(aid, {})
            if not scores:
                add(f"  {aid}  (no probe data in control run)")
                continue
            add(f"  {aid}")
            for trait, score in top_k(scores, top_k_n):
                add(_fmt_trait(trait, score))
            add("")

    # ── Steered agent delta ────────────────────────────────────────────────
    if control_agg and most_steered and most_steered in control_agg:
        add(_section(f"Effect on steered agent ({most_steered})"))
        steered_scores = steered_agg.get(most_steered, {})
        ctrl_scores = control_agg.get(most_steered, {})
        all_traits = set(steered_scores) | set(ctrl_scores)
        deltas = {
            t: steered_scores.get(t, 0.0) - ctrl_scores.get(t, 0.0)
            for t in all_traits
        }
        ranked = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k_n * 2]
        add(f"  {'Trait':<35} {'Steered':>8}  {'Control':>8}  {'Δ':>8}  {'':5}")
        add(f"  {'─'*35} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*5}")
        for trait, delta in ranked:
            sv = steered_scores.get(trait, 0.0)
            cv = ctrl_scores.get(trait, 0.0)
            sign = "+" if delta >= 0 else ""
            star = "★" if abs(delta) >= 0.05 else ""
            add(f"  {trait:<35} {sv:+8.3f}  {cv:+8.3f}  {sign}{delta:7.3f}  {star}")
        add("")

    # ── Cross-agent spill-over (do other agents change?) ──────────────────
    if control_agg and most_steered:
        add(_section("Cross-agent spill-over (unsteered agents)"))
        other_agents = [a for a in agents if a != most_steered]
        if not other_agents:
            add("  (no other agents to compare)")
        else:
            add(f"  Checking whether unsteered agents shift their trait projections.")
            significant_found = False
            for aid in other_agents:
                s_scores = steered_agg.get(aid, {})
                c_scores = control_agg.get(aid, {})
                if not s_scores or not c_scores:
                    continue
                big_shifts = [
                    (t, s_scores.get(t, 0) - c_scores.get(t, 0))
                    for t in set(s_scores) | set(c_scores)
                    if abs(s_scores.get(t, 0) - c_scores.get(t, 0)) >= 0.04
                ]
                if big_shifts:
                    significant_found = True
                    big_shifts.sort(key=lambda x: abs(x[1]), reverse=True)
                    add(f"\n  {aid} shows notable shifts (|Δ| ≥ 0.04):")
                    for trait, delta in big_shifts[:5]:
                        sign = "+" if delta >= 0 else ""
                        add(f"    {trait:<35} {sign}{delta:+.3f}")
            if not significant_found:
                add("  No significant trait shifts (|Δ| ≥ 0.04) detected in "
                    "unsteered agents.")
        add("")

    # ── Footer ────────────────────────────────────────────────────────────
    add("=" * 72)
    add("  ★ = |Δ| ≥ 0.05   Scores are cosine projections onto trait vectors.")
    add("  Positive = hidden state aligned with that trait direction.")
    add("=" * 72)

    output = "\n".join(lines)
    print(output)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"\nReport saved to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Qualitative persona probe analysis from episode JSONL logs."
    )
    ap.add_argument("--steered", required=True,
                    help="Path to the steered run's log directory.")
    ap.add_argument("--control", default=None,
                    help="Path to the control (noop) run's log directory "
                         "(optional, enables delta comparison).")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Number of top traits to show per agent (default 5).")
    ap.add_argument("--out", default=None,
                    help="Optional path to save the text report.")
    args = ap.parse_args()

    steered_records = load_run(args.steered)
    steered_summaries = load_summaries(args.steered)
    if not steered_records:
        print(f"ERROR: no probe data found in {args.steered}. "
              "Run with probing.enabled: true in the YAML first.", file=sys.stderr)
        sys.exit(1)

    control_records = None
    if args.control:
        control_records = load_run(args.control)
        if not control_records:
            print(f"WARNING: no probe data found in {args.control}. "
                  "Proceeding without control comparison.", file=sys.stderr)
            control_records = None

    report(steered_records, control_records, steered_summaries,
           top_k_n=args.top_k, out_path=args.out)


if __name__ == "__main__":
    main()
