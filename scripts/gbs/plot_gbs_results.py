"""Plot Group Belief Selection (GBS) experiment results across conditions.

Usage:
    python scripts/gbs/plot_gbs_results.py [--log-dir logs] [--out plots/gbs]
    python scripts/gbs/plot_gbs_results.py --log-dir logs --conditions gbs_exp_noop gbs_exp_activation_one

Finds all gbs_* run directories under --log-dir and produces:
    convergence_rate.png     — fraction of episodes where direction=="correct" per condition
    error_distribution.png  — final |error| boxplot by condition
    error_trajectory.png    — mean ± std of |error| per round (from JSONL, skipped if absent)
    rounds_to_convergence.png — distribution of rounds taken among converged episodes
    overview.png             — 4-panel summary
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COND_COLORS = [
    "#4C9BE8", "#E8604C", "#4CAF7D", "#F5A623",
    "#9B59B6", "#1ABC9C", "#E74C3C", "#95A5A6",
]
GRID_COLOR = "#DDDDDD"

# How to display condition names (strip gbs_exp_ prefix)
def _label(cond: str) -> str:
    label = re.sub(r"^gbs_exp_", "", cond)
    return label.replace("_", " ")


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_summaries(log_dir: str, conditions: list[str] | None = None) -> dict[str, list[dict]]:
    """Return {condition_name: [final_info, ...]}."""
    data: dict[str, list[dict]] = {}
    if conditions:
        run_dirs = [(c, os.path.join(log_dir, c)) for c in conditions]
    else:
        # auto-discover gbs_* dirs
        run_dirs = []
        for d in sorted(glob.glob(os.path.join(log_dir, "gbs_*"))):
            if os.path.isdir(d):
                run_dirs.append((os.path.basename(d), d))

    for cond, run_dir in run_dirs:
        eps = []
        for path in sorted(glob.glob(os.path.join(run_dir, "episode_*.summary.json"))):
            try:
                with open(path) as f:
                    s = json.load(f)
                info = s.get("final_info", {})
                if "direction" in info or "error" in info:
                    eps.append(info)
            except (json.JSONDecodeError, OSError):
                continue
        if eps:
            data[cond] = eps
    return data


def load_error_trajectories(log_dir: str, conditions: list[str]) -> dict[str, list[list[float]]]:
    """Return {cond: [[|error|_round0, |error|_round1, ...], ...]} from JSONL.

    Each JSONL line is one agent's step for a given turn. We deduplicate by turn
    (GBS is simultaneous — all N agents produce the same info dict at turn T).
    """
    data: dict[str, list[list[float]]] = {}
    for cond in conditions:
        run_dir = os.path.join(log_dir, cond)
        seqs = []
        for jpath in sorted(glob.glob(os.path.join(run_dir, "episode_*.jsonl"))):
            seen_turns: dict[int, float] = {}  # turn → |error|
            try:
                with open(jpath) as f:
                    for line in f:
                        step = json.loads(line.strip())
                        turn  = step.get("turn", 0)
                        error = step.get("info", {}).get("error", None)
                        if error is not None and turn not in seen_turns:
                            seen_turns[turn] = abs(error)
            except (json.JSONDecodeError, OSError):
                continue
            if seen_turns:
                seq = [seen_turns[t] for t in sorted(seen_turns)]
                seqs.append(seq)
        if seqs:
            data[cond] = seqs
    return data


# ── plot 1: convergence rate ──────────────────────────────────────────────────

def plot_convergence_rate(data: dict, out: str):
    conds = list(data)
    rates, ns, ci = [], [], []
    for cond in conds:
        eps   = data[cond]; t = len(eps) or 1
        conv  = sum(1 for e in eps if e.get("direction") == "correct")
        p     = conv / t
        z     = 1.96
        half  = z * (p * (1 - p) / t) ** 0.5
        rates.append(p * 100); ns.append(t); ci.append(half * 100)

    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(max(7, 2 * len(conds)), 5))
    colors = [COND_COLORS[i % len(COND_COLORS)] for i in range(len(conds))]
    ax.bar(x, rates, color=colors, zorder=3,
           yerr=ci, capsize=5, error_kw={"linewidth": 1.2})
    for i, (r, n) in enumerate(zip(rates, ns)):
        ax.text(i, r + ci[i] + 2, f"{r:.0f}%\n(n={n})", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(c) for c in conds], rotation=20, ha="right")
    ax.set_ylim(0, 115)
    _style(ax, "Convergence Rate by Condition", "Condition", "Episodes converged (%)")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 2: final error distribution ─────────────────────────────────────────

def plot_error_distribution(data: dict, out: str):
    conds  = list(data)
    errors = [[abs(e.get("error", 0)) for e in data[c]] for c in conds]
    colors = [COND_COLORS[i % len(COND_COLORS)] for i in range(len(conds))]

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(conds)), 5))
    rng = np.random.default_rng(42)
    bp  = ax.boxplot(
        errors, positions=range(len(conds)), widths=0.45,
        patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3,
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, (errs, col) in enumerate(zip(errors, colors)):
        jit = rng.uniform(-0.15, 0.15, len(errs))
        ax.scatter(np.full(len(errs), i) + jit, errs,
                   color=col, alpha=0.4, s=12, zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.7)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([_label(c) for c in conds], rotation=20, ha="right")
    _style(ax, "Final |Error| Distribution by Condition",
           "Condition", "Final |error| from target")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 3: error trajectory ──────────────────────────────────────────────────

def plot_error_trajectory(trajectories: dict, out: str):
    if not trajectories:
        print("No JSONL data — skipping error_trajectory.png"); return
    conds  = list(trajectories)
    colors = [COND_COLORS[i % len(COND_COLORS)] for i in range(len(conds))]

    fig, ax = plt.subplots(figsize=(9, 5))
    for cond, col in zip(conds, colors):
        seqs    = trajectories[cond]
        max_len = max(len(s) for s in seqs)
        padded  = np.array([s + [s[-1]] * (max_len - len(s)) for s in seqs], dtype=float)
        mean    = padded.mean(axis=0)
        std     = padded.std(axis=0)
        rounds  = np.arange(max_len)
        ax.plot(rounds, mean, color=col, linewidth=2,
                label=f"{_label(cond)} (n={len(seqs)})")
        ax.fill_between(rounds, np.maximum(mean - std, 0), mean + std,
                        color=col, alpha=0.15)
    ax.axhline(0, color="#2ECC71", linewidth=1, linestyle="--", label="Convergence (0)")
    _style(ax, "Mean |Error| Over Rounds", "Round", "Mean |error|")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 4: rounds to convergence ─────────────────────────────────────────────

def plot_rounds_to_convergence(trajectories: dict, data: dict, out: str):
    """Use JSONL trajectories if available, else fall back to max_rounds from summary."""
    conds = list(data)

    # Try to read rounds from JSONL (trajectory length of converged episodes)
    rounds_by_cond: dict[str, list[int]] = {}
    if trajectories:
        for cond in conds:
            if cond not in trajectories:
                continue
            eps_info = data[cond]
            seqs     = trajectories[cond]
            # Only converged episodes
            conv_rounds = [len(s) for s, e in zip(seqs, eps_info)
                           if e.get("direction") == "correct"]
            if conv_rounds:
                rounds_by_cond[cond] = conv_rounds
    if not rounds_by_cond:
        print("No trajectory data for rounds_to_convergence — skipping."); return

    conds_with  = [c for c in conds if c in rounds_by_cond]
    colors      = [COND_COLORS[i % len(COND_COLORS)] for i, c in enumerate(conds) if c in rounds_by_cond]

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(conds_with)), 5))
    rng = np.random.default_rng(0)
    bp  = ax.boxplot(
        [rounds_by_cond[c] for c in conds_with],
        positions=range(len(conds_with)), widths=0.45,
        patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3,
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, (c, col) in enumerate(zip(conds_with, colors)):
        rs  = rounds_by_cond[c]
        jit = rng.uniform(-0.15, 0.15, len(rs))
        ax.scatter(np.full(len(rs), i) + jit, rs,
                   color=col, alpha=0.4, s=12, zorder=4)
        ax.text(i, max(rs) + 0.5, f"n={len(rs)}", ha="center", fontsize=8)
    ax.set_xticks(range(len(conds_with)))
    ax.set_xticklabels([_label(c) for c in conds_with], rotation=20, ha="right")
    _style(ax, "Rounds to Convergence (converged episodes only)",
           "Condition", "Rounds")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir",    default="logs",
                    help="Parent directory containing gbs_* run directories")
    ap.add_argument("--out",        default="plots/gbs")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Specific run_id names to include (default: auto-discover gbs_*)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = load_summaries(args.log_dir, args.conditions)
    if not data:
        print("No GBS summaries found. Run episodes first."); return

    for cond, eps in sorted(data.items()):
        conv = sum(1 for e in eps if e.get("direction") == "correct")
        mean_err = np.mean([abs(e.get("error", 0)) for e in eps])
        print(f"  {cond}: {len(eps)} eps — converged {conv} ({conv/len(eps)*100:.0f}%), "
              f"mean |error| {mean_err:.1f}")

    trajectories = load_error_trajectories(args.log_dir, list(data))

    plot_convergence_rate(data,       os.path.join(args.out, "convergence_rate.png"))
    plot_error_distribution(data,     os.path.join(args.out, "error_distribution.png"))
    plot_error_trajectory(trajectories, os.path.join(args.out, "error_trajectory.png"))
    plot_rounds_to_convergence(trajectories, data,
                               os.path.join(args.out, "rounds_to_convergence.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
