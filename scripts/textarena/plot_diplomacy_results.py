"""Plot Diplomacy experiment results.

Usage:
    python scripts/textarena/plot_diplomacy_results.py [--log-dir logs/diplomacy] [--out plots/diplomacy]

Reads episode_*.summary.json from logs/diplomacy/diplomacy_noop_5p/ and produces:
    reward_distribution.png  — per-player reward distribution across episodes
    win_rate.png             — fraction of episodes each player position wins
    reward_over_episodes.png — cumulative average reward per player (learning/stability check)
    overview.png             — 2-panel summary
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

PLAYER_COLORS = [
    "#4C9BE8", "#E8604C", "#4CAF7D", "#F5A623", "#9B59B6",
    "#1ABC9C", "#E74C3C",
]
GRID_COLOR = "#DDDDDD"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_summaries(log_dir: str) -> list[dict]:
    summaries = []
    for path in sorted(glob.glob(os.path.join(log_dir, "**", "episode_*.summary.json"),
                                 recursive=True)):
        try:
            with open(path) as f:
                s = json.load(f)
            if s.get("final_rewards"):
                summaries.append(s)
        except (json.JSONDecodeError, OSError):
            continue
    return summaries


def _player_order(summaries: list[dict]) -> list[str]:
    if not summaries:
        return []
    pids = list(summaries[0].get("final_rewards", {}).keys())
    try:
        return sorted(pids, key=lambda p: int(re.search(r"\d+", p).group()))
    except (AttributeError, TypeError):
        return sorted(pids)


# ── plot 1: reward distribution by player ────────────────────────────────────

def plot_reward_distribution(summaries: list, out: str):
    players = _player_order(summaries)
    if not players:
        return
    rewards_by_player = {p: [s["final_rewards"].get(p, 0) for s in summaries] for p in players}

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(players)), 5))
    rng = np.random.default_rng(42)
    bp  = ax.boxplot(
        [rewards_by_player[p] for p in players],
        positions=range(len(players)), widths=0.45,
        patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3,
    )
    for i, (patch, p) in enumerate(zip(bp["boxes"], players)):
        col = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, p in enumerate(players):
        col = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        rs  = rewards_by_player[p]
        jit = rng.uniform(-0.15, 0.15, len(rs))
        ax.scatter(np.full(len(rs), i) + jit, rs,
                   color=col, alpha=0.45, s=14, zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels([f"P{i}\n({p})" for i, p in enumerate(players)])
    _style(ax, f"Reward Distribution by Player (n={len(summaries)} episodes)",
           "Player position", "Final reward")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 2: win rate by player ────────────────────────────────────────────────

def _is_winner(reward: float, all_rewards: list[float]) -> bool:
    if not all_rewards:
        return False
    return reward == max(all_rewards)


def plot_win_rate(summaries: list, out: str):
    players = _player_order(summaries)
    if not players:
        return
    wins = defaultdict(int)
    total = len(summaries)
    for s in summaries:
        rews = s["final_rewards"]
        all_r = list(rews.values())
        for p in players:
            if _is_winner(rews.get(p, float("-inf")), all_r):
                wins[p] += 1

    win_rates = [wins[p] / total * 100 for p in players]
    ci        = [1.96 * (w/100 * (1 - w/100) / total) ** 0.5 * 100 for w in win_rates]
    uniform   = 100 / len(players)

    x = np.arange(len(players))
    fig, ax = plt.subplots(figsize=(max(7, 2 * len(players)), 5))
    colors = [PLAYER_COLORS[i % len(PLAYER_COLORS)] for i in range(len(players))]
    ax.bar(x, win_rates, color=colors, zorder=3,
           yerr=ci, capsize=5, error_kw={"linewidth": 1.2})
    ax.axhline(uniform, color="#AAAAAA", linewidth=1.2, linestyle="--",
               label=f"Uniform ({uniform:.0f}%)")
    for i, (wr, c) in enumerate(zip(win_rates, ci)):
        ax.text(i, wr + c + 1.5, f"{wr:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i}\n({p})" for i, p in enumerate(players)])
    ax.set_ylim(0, max(win_rates) * 1.3 + 5)
    _style(ax, f"Win Rate by Player Position (n={total} episodes)",
           "Player position", "Win rate (%)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 3: cumulative average reward ─────────────────────────────────────────

def plot_reward_over_episodes(summaries: list, out: str):
    players = _player_order(summaries)
    if not players or not summaries:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ep_idx  = np.arange(1, len(summaries) + 1)
    for i, p in enumerate(players):
        col  = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        rews = np.array([s["final_rewards"].get(p, 0) for s in summaries])
        cumavg = np.cumsum(rews) / ep_idx
        ax.plot(ep_idx, cumavg, color=col, linewidth=1.5, label=p)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    _style(ax, "Cumulative Average Reward per Player",
           "Episode", "Cumulative mean reward")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 4: overview (2-panel) ────────────────────────────────────────────────

def plot_overview(summaries: list, out: str):
    players = _player_order(summaries)
    if not players:
        return
    total = len(summaries)

    wins = defaultdict(int)
    rewards_by_player = {p: [] for p in players}
    for s in summaries:
        rews = s["final_rewards"]
        all_r = list(rews.values())
        for p in players:
            r = rews.get(p, 0)
            rewards_by_player[p].append(r)
            if _is_winner(r, all_r):
                wins[p] += 1

    win_rates = [wins[p] / total * 100 for p in players]
    uniform   = 100 / len(players)
    colors    = [PLAYER_COLORS[i % len(PLAYER_COLORS)] for i in range(len(players))]
    x         = np.arange(len(players))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Diplomacy Baseline Results (n={total})", fontsize=14, fontweight="bold")

    ax = axes[0]
    rng = np.random.default_rng(0)
    bp = ax.boxplot(
        [rewards_by_player[p] for p in players],
        positions=range(len(players)), widths=0.45,
        patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3,
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, (p, col) in enumerate(zip(players, colors)):
        jit = rng.uniform(-0.1, 0.1, len(rewards_by_player[p]))
        ax.scatter(np.full(len(rewards_by_player[p]), i) + jit, rewards_by_player[p],
                   color=col, alpha=0.4, s=10, zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(players))); ax.set_xticklabels([f"P{i}" for i in range(len(players))])
    _style(ax, "Reward Distribution", "Player", "Final reward")

    ax = axes[1]
    ax.bar(x, win_rates, color=colors, zorder=3)
    ax.axhline(uniform, color="#AAAAAA", linewidth=1.2, linestyle="--",
               label=f"Uniform ({uniform:.0f}%)")
    for i, wr in enumerate(win_rates):
        ax.text(i, wr + 1.5, f"{wr:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"P{i}" for i in range(len(players))])
    ax.set_ylim(0, max(win_rates + [uniform]) * 1.35 + 5)
    _style(ax, "Win Rate by Player Position", "Player", "Win rate (%)")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/diplomacy")
    ap.add_argument("--out",     default="plots/diplomacy")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summaries = load_summaries(args.log_dir)
    if not summaries:
        print("No summaries found. Run episodes first."); return

    players = _player_order(summaries)
    print(f"Loaded {len(summaries)} episodes, {len(players)} players: {players}")
    for p in players:
        mean_r = np.mean([s["final_rewards"].get(p, 0) for s in summaries])
        print(f"  {p}: mean reward {mean_r:.3f}")

    plot_reward_distribution(summaries,  os.path.join(args.out, "reward_distribution.png"))
    plot_win_rate(summaries,             os.path.join(args.out, "win_rate.png"))
    plot_reward_over_episodes(summaries, os.path.join(args.out, "reward_over_episodes.png"))
    plot_overview(summaries,             os.path.join(args.out, "overview.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
