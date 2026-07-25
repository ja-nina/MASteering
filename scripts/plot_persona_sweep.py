"""Plot results of the behavioural-persona impact sweep.

Reads episode summary files from logs/persona_sweep/ and produces:
  persona_success_rate.png   — convergence rate per persona, sorted
  persona_rounds.png         — mean rounds-to-success per persona (converged episodes only)
  persona_overview.png       — combined 2-panel figure

Usage
-----
python scripts/plot_persona_sweep.py [--log-dir logs/persona_sweep/]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_ID_RE = re.compile(r"persona_impact_(?P<condition>[^_]+(?:_[^_]+)*)_2p_14b")

CONDITION_ORDER = [
    "plain",
    "cooperative", "helpful", "mediator", "social",
    "analytical", "scientist", "game_theorist",
    "optimistic", "pessimistic",
    "confident", "uncertain", "overconfident",
    "risk_averse", "risk_seeking",
    "leader", "competitive",
    "contrarian", "intuitive", "chaotic",
    "sycophantic",
]

CONDITION_LABEL = {c: c.replace("_", "-") for c in CONDITION_ORDER}
CONDITION_LABEL["game_theorist"] = "game-theorist"
CONDITION_LABEL["risk_averse"]   = "risk-averse"
CONDITION_LABEL["risk_seeking"]  = "risk-seeking"


def load_summaries(log_dir: str) -> dict[str, list[dict]]:
    """Return {condition: [summary_dict, ...]} from all summary JSON files."""
    data: dict[str, list[dict]] = {}
    root = Path(log_dir)
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_ID_RE.match(run_dir.name)
        if not m:
            continue
        condition = m.group("condition")
        summaries = []
        for path in sorted(run_dir.glob("episode_*.summary.json")):
            with open(path, encoding="utf-8") as f:
                summaries.append(json.load(f))
        if summaries:
            data[condition] = summaries
    return data


def compute_metrics(summaries: list[dict]) -> tuple[float, float, int]:
    """Return (success_rate, mean_rounds_if_converged, n_episodes)."""
    n = len(summaries)
    if n == 0:
        return 0.0, float("nan"), 0
    converged = [s for s in summaries if s.get("gbs_converged", False)]
    rate = len(converged) / n
    if converged:
        mean_rounds = float(np.mean([s["gbs_converged_round"] for s in converged]))
    else:
        mean_rounds = float("nan")
    return rate, mean_rounds, n


def plot_bar(ax, conditions, values, *, ylabel, title, baseline_val=None,
             color="steelblue", baseline_color="crimson"):
    labels = [CONDITION_LABEL.get(c, c) for c in conditions]
    x = np.arange(len(conditions))
    bars = ax.bar(x, values, color=color, edgecolor="white", linewidth=0.5)
    if baseline_val is not None:
        ax.axhline(baseline_val, color=baseline_color, linestyle="--",
                   linewidth=1.2, label="plain baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if baseline_val is not None:
        ax.legend(fontsize=8)
    # Annotate n on top of each bar
    for bar, v in zip(bars, values):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/persona_sweep/")
    parser.add_argument("--out-dir", default="plots/persona_sweep/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = load_summaries(args.log_dir)

    if not data:
        print(f"No summary files found under {args.log_dir}")
        return

    # Preserve CONDITION_ORDER for known conditions; append unknowns at end
    known   = [c for c in CONDITION_ORDER if c in data]
    unknown = [c for c in data if c not in CONDITION_ORDER]
    conditions = known + unknown

    rates  = []
    rounds = []
    ns     = []
    for c in conditions:
        r, rnd, n = compute_metrics(data[c])
        rates.append(r)
        rounds.append(rnd)
        ns.append(n)

    baseline_rate   = rates[conditions.index("plain")]   if "plain" in conditions else None
    baseline_rounds = rounds[conditions.index("plain")]  if "plain" in conditions else None

    # ── Figure 1: success rate ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    plot_bar(ax, conditions, rates,
             ylabel="convergence rate", title="Behavioural persona — convergence rate (100 episodes)",
             baseline_val=baseline_rate)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out = os.path.join(args.out_dir, "persona_success_rate.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)

    # ── Figure 2: rounds to success ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    plot_bar(ax, conditions, rounds,
             ylabel="mean rounds (converged)", title="Behavioural persona — rounds to convergence",
             baseline_val=baseline_rounds, color="teal")
    fig.tight_layout()
    out = os.path.join(args.out_dir, "persona_rounds.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)

    # ── Figure 3: combined overview ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    plot_bar(axes[0], conditions, rates,
             ylabel="convergence rate", title="Persona impact on coordination (Qwen3-14B, 2 players, 100 ep)",
             baseline_val=baseline_rate)
    axes[0].set_ylim(0, 1.05)
    plot_bar(axes[1], conditions, rounds,
             ylabel="mean rounds (converged)", title="",
             baseline_val=baseline_rounds, color="teal")
    fig.tight_layout()
    out = os.path.join(args.out_dir, "persona_overview.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)

    # ── Console summary ─────────────────────────────────────────────────────
    print(f"\n{'Condition':<20} {'N':>5} {'Converged':>10} {'Rate':>7} {'AvgRounds':>10}")
    print("-" * 58)
    for c, r, rnd, n in zip(conditions, rates, rounds, ns):
        converged_n = int(round(r * n))
        rnd_s = f"{rnd:.1f}" if not np.isnan(rnd) else "—"
        marker = " ←" if c == "plain" else ""
        print(f"{c:<20} {n:>5} {converged_n:>10} {r:>7.1%} {rnd_s:>10}{marker}")


if __name__ == "__main__":
    main()
