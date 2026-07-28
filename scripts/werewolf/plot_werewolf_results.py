"""Plot werewolf experiment results across player counts.

Usage:
    python scripts/werewolf/plot_werewolf_results.py [--log-dir logs/werewolf] [--out plots/werewolf]

Reads all episode_*.summary.json files under log_dir/werewolf_noop_{N}p/ for N in 6..10
(or whatever run dirs are present) and produces:
    win_rates.png          — village vs werewolf win rate by player count
    game_length.png        — distribution of game duration (days) by player count
    eliminations.png       — avg wolves killed / villagers killed by player count & winner
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── data loading ─────────────────────────────────────────────────────────────

def load_summaries(log_dir: str) -> Dict[int, List[dict]]:
    """Return {num_players: [final_info, ...]} from all matching run dirs."""
    pattern = re.compile(r"werewolf_noop_(\d+)p$")
    data: Dict[int, List[dict]] = defaultdict(list)

    for run_dir in sorted(glob.glob(os.path.join(log_dir, "werewolf_noop_*p"))):
        m = pattern.search(run_dir)
        if not m:
            continue
        n = int(m.group(1))
        for path in glob.glob(os.path.join(run_dir, "episode_*.summary.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    summary = json.load(f)
                info = summary.get("final_info", {})
                if info and "winner" in info:
                    data[n].append(info)
            except (json.JSONDecodeError, OSError):
                continue

    return dict(data)


# ── plot helpers ──────────────────────────────────────────────────────────────

VILLAGE_COLOR = "#4C9BE8"
WOLF_COLOR    = "#E8604C"
GRID_COLOR    = "#DDDDDD"

def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── plot 1: win rates ─────────────────────────────────────────────────────────

def plot_win_rates(data: Dict[int, List[dict]], out_path: str) -> None:
    counts = sorted(data)
    village_rates, wolf_rates, n_eps = [], [], []

    for n in counts:
        episodes = data[n]
        total = len(episodes)
        village = sum(1 for e in episodes if e.get("winner") == "village")
        village_rates.append(village / total if total else 0.0)
        wolf_rates.append((total - village) / total if total else 0.0)
        n_eps.append(total)

    x = np.arange(len(counts))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_v = ax.bar(x - w / 2, [r * 100 for r in village_rates], w,
                    label="Village", color=VILLAGE_COLOR, zorder=3)
    bars_w = ax.bar(x + w / 2, [r * 100 for r in wolf_rates], w,
                    label="Werewolf", color=WOLF_COLOR, zorder=3)

    for bar, rate, n in zip(bars_v, village_rates, n_eps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate*100:.0f}%", ha="center", va="bottom", fontsize=9)
    for bar, rate in zip(bars_w, wolf_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate*100:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}p\n(n={ep})" for n, ep in zip(counts, n_eps)])
    ax.set_ylim(0, 115)
    ax.axhline(50, color="#AAAAAA", linewidth=1, linestyle="--", zorder=2)
    _style_ax(ax, "Win Rate by Player Count", "Players", "Win Rate (%)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── plot 2: game length distribution ─────────────────────────────────────────

def plot_game_length(data: Dict[int, List[dict]], out_path: str) -> None:
    counts = sorted(data)
    days_by_n = [[e.get("days", 0) for e in data[n]] for n in counts]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        days_by_n,
        positions=range(len(counts)),
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor="#AACCEE", color="#5588AA"),
        whiskerprops=dict(color="#5588AA"),
        capprops=dict(color="#5588AA"),
        flierprops=dict(marker="o", markerfacecolor="#5588AA", markersize=4, alpha=0.5),
        zorder=3,
    )

    # overlay individual points (jittered)
    rng = np.random.default_rng(42)
    for i, days in enumerate(days_by_n):
        jitter = rng.uniform(-0.15, 0.15, size=len(days))
        ax.scatter(np.full(len(days), i) + jitter, days,
                   color=VILLAGE_COLOR, alpha=0.5, s=18, zorder=4)

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{n}p" for n in counts])
    _style_ax(ax, "Game Length Distribution by Player Count", "Players", "Days")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── plot 3: eliminations by winner ───────────────────────────────────────────

def plot_eliminations(data: Dict[int, List[dict]], out_path: str) -> None:
    counts = sorted(data)

    # for each player count: avg wolves/villagers killed, split by who won
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, winner, color, title in [
        (axes[0], "village", VILLAGE_COLOR, "Village Wins"),
        (axes[1], "werewolf", WOLF_COLOR,   "Werewolf Wins"),
    ]:
        avg_wolves, avg_village, ns = [], [], []
        for n in counts:
            eps = [e for e in data[n] if e.get("winner") == winner]
            ns.append(len(eps))
            avg_wolves.append(np.mean([e.get("wolves_killed", 0) for e in eps]) if eps else 0)
            avg_village.append(np.mean([e.get("village_killed", 0) for e in eps]) if eps else 0)

        x = np.arange(len(counts))
        w = 0.35
        ax.bar(x - w / 2, avg_wolves, w, label="Wolves killed", color=WOLF_COLOR, zorder=3)
        ax.bar(x + w / 2, avg_village, w, label="Villagers killed", color=VILLAGE_COLOR, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}p\n(n={ep})" for n, ep in zip(counts, ns)])
        _style_ax(ax, f"Avg Eliminations — {title}", "Players", "Avg players eliminated")
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── plot 4: elimination role breakdown (stacked by day) ──────────────────────

def plot_role_breakdown(data: Dict[int, List[dict]], out_path: str) -> None:
    """Stacked bar: by-role share of eliminations for each player count."""
    counts = sorted(data)
    role_colors = {
        "Werewolf": WOLF_COLOR,
        "Villager": "#7FC97F",
        "Seer":     "#FDC086",
        "Doctor":   "#BEAED4",
    }
    all_roles = ["Villager", "Seer", "Doctor", "Werewolf"]

    counts_by_role: Dict[str, List[float]] = {r: [] for r in all_roles}
    totals = []

    for n in counts:
        role_counts: Dict[str, int] = defaultdict(int)
        total_elims = 0
        for ep in data[n]:
            for entry in ep.get("elimination_log", []):
                role = entry.get("role", "Villager")
                role_counts[role] += 1
                total_elims += 1
        totals.append(total_elims)
        for r in all_roles:
            counts_by_role[r].append(role_counts.get(r, 0) / total_elims if total_elims else 0)

    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(counts))

    for role in all_roles:
        vals = np.array(counts_by_role[role]) * 100
        ax.bar(x, vals, bottom=bottom, label=role,
               color=role_colors[role], zorder=3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}p" for n in counts])
    ax.set_ylim(0, 105)
    _style_ax(ax, "Role Share of All Eliminations", "Players", "% of total eliminations")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot werewolf results across player counts")
    parser.add_argument("--log-dir", default="logs/werewolf",
                        help="Root dir containing werewolf_noop_Np sub-dirs")
    parser.add_argument("--out", default="plots/werewolf",
                        help="Output directory for PNG files")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading summaries from {args.log_dir} ...")
    data = load_summaries(args.log_dir)
    if not data:
        print("No summary files found. Run some episodes first.")
        return

    for n, eps in sorted(data.items()):
        winners = [e.get("winner") for e in eps]
        v = sum(1 for w in winners if w == "village")
        ww = sum(1 for w in winners if w == "werewolf")
        print(f"  {n}p: {len(eps)} episodes — village {v}, werewolf {ww}")

    plot_win_rates(data,    os.path.join(args.out, "win_rates.png"))
    plot_game_length(data,  os.path.join(args.out, "game_length.png"))
    plot_eliminations(data, os.path.join(args.out, "eliminations.png"))
    plot_role_breakdown(data, os.path.join(args.out, "role_breakdown.png"))

    print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main()
