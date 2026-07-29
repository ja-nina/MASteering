"""Plot Hanabi experiment results across player counts.

Usage:
    python scripts/hanabi/plot_hanabi_results.py [--log-dir logs/hanabi] [--out plots/hanabi]

Reads episode_*.summary.json (and optionally episode_*.jsonl for step-level data)
from logs/hanabi/hanabi_noop_{N}p/ for N in 2..5 and produces:
    score_distribution.png   — score boxplot + points by player count
    terminal_conditions.png  — bomb / deck-exhausted / perfect breakdown by player count
    score_histogram.png      — overlapping score histograms across player counts
    score_progression.png    — mean score over turns (from JSONL, skipped if absent)
    overview.png             — 3-panel summary
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

PLAYER_COLORS = ["#4C9BE8", "#E8604C", "#4CAF7D", "#F5A623"]
GRID_COLOR    = "#DDDDDD"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_summaries(log_dir: str) -> dict[int, list[dict]]:
    pattern = re.compile(r"hanabi_noop_(\d+)p$")
    data: dict[int, list[dict]] = defaultdict(list)
    for run_dir in sorted(glob.glob(os.path.join(log_dir, "hanabi_noop_*p"))):
        m = pattern.search(run_dir)
        if not m:
            continue
        n = int(m.group(1))
        for path in sorted(glob.glob(os.path.join(run_dir, "episode_*.summary.json"))):
            try:
                with open(path) as f:
                    s = json.load(f)
                info = s.get("final_info", {})
                if "score" in info:
                    data[n].append(info)
            except (json.JSONDecodeError, OSError):
                continue
    return dict(data)


def load_score_progressions(log_dir: str) -> dict[int, list[list[float]]]:
    """Return {num_players: [[score_at_turn_0, score_at_turn_1, ...], ...]} from JSONL."""
    pattern = re.compile(r"hanabi_noop_(\d+)p$")
    data: dict[int, list[list[float]]] = defaultdict(list)
    for run_dir in sorted(glob.glob(os.path.join(log_dir, "hanabi_noop_*p"))):
        m = pattern.search(run_dir)
        if not m:
            continue
        n = int(m.group(1))
        for jpath in sorted(glob.glob(os.path.join(run_dir, "episode_*.jsonl"))):
            scores = []
            last_score = 0
            last_turn  = -1
            try:
                with open(jpath) as f:
                    for line in f:
                        step = json.loads(line.strip())
                        turn  = step.get("turn", 0)
                        score = step.get("info", {}).get("score", last_score)
                        if turn != last_turn:
                            scores.append(score)
                            last_score = score
                            last_turn  = turn
            except (json.JSONDecodeError, OSError):
                continue
            if scores:
                data[n].append(scores)
    return dict(data)


# ── plot 1: score distribution ────────────────────────────────────────────────

def plot_score_distribution(data: dict, out: str):
    counts = sorted(data)
    scores_by_n = [[e["score"] for e in data[n]] for n in counts]

    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.random.default_rng(42)
    bp = ax.boxplot(
        scores_by_n, positions=range(len(counts)), widths=0.45,
        patch_artist=True, medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor="#D0E8F5", color="#3A80C0"),
        whiskerprops=dict(color="#3A80C0"), capprops=dict(color="#3A80C0"),
        flierprops=dict(marker="o", markerfacecolor="#3A80C0", markersize=3, alpha=0.4),
        zorder=3,
    )
    for i, (scores, col) in enumerate(zip(scores_by_n, PLAYER_COLORS)):
        jit = rng.uniform(-0.15, 0.15, len(scores))
        ax.scatter(np.full(len(scores), i) + jit, scores,
                   color=col, alpha=0.55, s=16, zorder=4)
    ax.axhline(25, color="#2ECC71", linewidth=1.2, linestyle="--", label="Perfect (25)")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{n}p\n(n={len(data[n])})" for n in counts])
    ax.set_ylim(-1, 28)
    _style(ax, "Final Score Distribution by Player Count", "Players", "Score (max 25)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 2: terminal condition breakdown ──────────────────────────────────────

def _terminal_condition(info: dict) -> str:
    if info.get("fuse_tokens", 1) <= 0:
        return "bomb"
    if info.get("score", 0) >= 25:
        return "perfect"
    return "deck"


def plot_terminal_conditions(data: dict, out: str):
    counts = sorted(data)
    bomb_r, deck_r, perf_r = [], [], []
    for n in counts:
        eps = data[n]; t = len(eps) or 1
        conds = [_terminal_condition(e) for e in eps]
        bomb_r.append(conds.count("bomb") / t * 100)
        deck_r.append(conds.count("deck") / t * 100)
        perf_r.append(conds.count("perfect") / t * 100)

    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, bomb_r, color="#E8604C", label="Bomb (fuse=0)",       zorder=3)
    ax.bar(x, deck_r, color="#4C9BE8", label="Deck exhausted",       zorder=3, bottom=bomb_r)
    bot = [a + b for a, b in zip(bomb_r, deck_r)]
    ax.bar(x, perf_r, color="#2ECC71", label="Perfect score (25)",   zorder=3, bottom=bot)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}p\n(n={len(data[n])})" for n in counts])
    ax.set_ylim(0, 110)
    _style(ax, "Terminal Condition by Player Count", "Players", "% of episodes")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 3: score histogram ───────────────────────────────────────────────────

def plot_score_histogram(data: dict, out: str):
    counts = sorted(data)
    fig, axes = plt.subplots(1, len(counts), figsize=(4 * len(counts), 4), sharey=True)
    if len(counts) == 1:
        axes = [axes]

    for ax, n, col in zip(axes, counts, PLAYER_COLORS):
        scores = [e["score"] for e in data[n]]
        bins = np.arange(-0.5, 26.5, 1)
        ax.hist(scores, bins=bins, color=col, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(np.mean(scores), color="black", linewidth=1.5, linestyle="--",
                   label=f"mean={np.mean(scores):.1f}")
        ax.set_title(f"{n} players", fontsize=11, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_xlim(-1, 26)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("Episodes")
    fig.suptitle("Score Histograms by Player Count", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 4: score progression over turns (JSONL) ─────────────────────────────

def plot_score_progression(progressions: dict, out: str):
    if not progressions:
        print("No JSONL data found — skipping score_progression.png")
        return
    counts = sorted(progressions)
    fig, ax = plt.subplots(figsize=(9, 5))
    for n, col in zip(counts, PLAYER_COLORS):
        seqs = progressions[n]
        max_len = max(len(s) for s in seqs)
        # pad short sequences with last value
        padded = np.array([s + [s[-1]] * (max_len - len(s)) for s in seqs], dtype=float)
        mean   = padded.mean(axis=0)
        std    = padded.std(axis=0)
        turns  = np.arange(max_len)
        ax.plot(turns, mean, color=col, linewidth=2, label=f"{n}p (n={len(seqs)})")
        ax.fill_between(turns, mean - std, mean + std, color=col, alpha=0.15)
    ax.axhline(25, color="#2ECC71", linewidth=1, linestyle="--", label="Perfect (25)")
    _style(ax, "Score Progression Over Turns", "Turn", "Score")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 5: fuse usage ────────────────────────────────────────────────────────

def plot_fuse_usage(data: dict, out: str):
    counts = sorted(data)
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(0)
    for i, (n, col) in enumerate(zip(counts, PLAYER_COLORS)):
        fuses = [e.get("fuse_tokens", 3) for e in data[n]]
        jit   = rng.uniform(-0.15, 0.15, len(fuses))
        ax.scatter(np.full(len(fuses), i) + jit, fuses,
                   color=col, alpha=0.5, s=18, zorder=3, label=f"{n}p")
        ax.errorbar(i, np.mean(fuses), yerr=np.std(fuses),
                    color="black", linewidth=1.5, capsize=5, zorder=4)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{n}p\n(n={len(data[n])})" for n in counts])
    ax.set_ylim(-0.2, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.axhline(0, color=GRID_COLOR, linewidth=0.7)
    _style(ax, "Fuse Tokens Remaining at Game End", "Players", "Fuse tokens (0 = bomb)")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/hanabi")
    ap.add_argument("--out",     default="plots/hanabi")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = load_summaries(args.log_dir)
    if not data:
        print("No summaries found. Run episodes first."); return

    for n, eps in sorted(data.items()):
        mean_s = np.mean([e["score"] for e in eps])
        bombs  = sum(1 for e in eps if e.get("fuse_tokens", 1) <= 0)
        print(f"  {n}p: {len(eps)} eps — mean score {mean_s:.1f}, bombs {bombs}")

    progressions = load_score_progressions(args.log_dir)

    plot_score_distribution(data,     os.path.join(args.out, "score_distribution.png"))
    plot_terminal_conditions(data,    os.path.join(args.out, "terminal_conditions.png"))
    plot_score_histogram(data,        os.path.join(args.out, "score_histogram.png"))
    plot_fuse_usage(data,             os.path.join(args.out, "fuse_usage.png"))
    plot_score_progression(progressions, os.path.join(args.out, "score_progression.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
