"""Plot SecretMafia experiment results.

Usage:
    python scripts/mafia/plot_mafia_results.py [--log-dir logs/mafia] [--out plots/mafia]

Reads episode_*.summary.json from logs/mafia/mafia_noop_8p/ and produces:
    win_rates.png          — mafia vs civilian win rate
    reward_distribution.png — per-player reward spread
    survival_rate.png      — how often each player position survives to end
    reward_over_episodes.png — cumulative average reward (stability / convergence check)
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

MAFIA_COLOR    = "#E8604C"
CIVILIAN_COLOR = "#4C9BE8"
GRID_COLOR     = "#DDDDDD"
PLAYER_COLORS  = [
    "#4C9BE8", "#E8604C", "#4CAF7D", "#F5A623",
    "#9B59B6", "#1ABC9C", "#E74C3C", "#95A5A6",
]


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
    pids = list(summaries[0]["final_rewards"].keys())
    try:
        return sorted(pids, key=lambda p: int(re.search(r"\d+", p).group()))
    except (AttributeError, TypeError):
        return sorted(pids)


def _winner(summary: dict) -> str:
    """Infer winner from close_info or reward pattern."""
    info = summary.get("final_info", {})
    # TextArena stores close_info inside final_info
    close = info.get("close_info", info)
    if isinstance(close, dict):
        w = close.get("winner", close.get("winning_team", ""))
        if w:
            return str(w).lower()
    # fallback: if any player has reward > 0, check majority
    rews = summary.get("final_rewards", {})
    if not rews:
        return "unknown"
    pos = sum(1 for v in rews.values() if v > 0)
    total = len(rews)
    # mafia typically wins with fewer players getting positive reward
    return "unknown"


# ── plot 1: win rates ─────────────────────────────────────────────────────────

def plot_win_rates(summaries: list, out: str):
    """
    TextArena SecretMafia rewards:
      winners (mafia OR civilians depending on who wins) get reward > 0.
    We infer the winning team from close_info['winner'] when available,
    and fall back to checking the reward structure.
    """
    mafia_wins = 0
    civ_wins   = 0
    unknown    = 0
    for s in summaries:
        w = _winner(s)
        if "mafia" in w:
            mafia_wins += 1
        elif w in ("civilian", "town", "good", "village"):
            civ_wins += 1
        else:
            unknown += 1

    total = len(summaries)
    fig, ax = plt.subplots(figsize=(7, 5))

    if mafia_wins + civ_wins > 0:
        labels = ["Mafia wins", "Civilian wins"]
        values = [mafia_wins / total * 100, civ_wins / total * 100]
        colors = [MAFIA_COLOR, CIVILIAN_COLOR]
        if unknown:
            labels.append("Unknown")
            values.append(unknown / total * 100)
            colors.append("#95A5A6")
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, zorder=3)
        for i, v in enumerate(values):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(values) * 1.3 + 10)
    else:
        # close_info winner field not present — show raw reward positivity
        players = _player_order(summaries)
        frac_pos = [
            sum(1 for s in summaries if s["final_rewards"].get(p, 0) > 0) / total * 100
            for p in players
        ]
        x = np.arange(len(players))
        ax.bar(x, frac_pos, color=PLAYER_COLORS[:len(players)], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(players, rotation=20, ha="right")
        ax.set_ylabel("% episodes with positive reward")
        ax.set_ylim(0, 115)
        ax.axhline(100 / len(players), color="#AAAAAA", linewidth=1,
                   linestyle="--", label="Uniform")
        ax.legend(frameon=False)

    _style(ax, f"Win Rates (n={total} episodes)", "Outcome", "% of episodes")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 2: reward distribution per player ────────────────────────────────────

def plot_reward_distribution(summaries: list, out: str):
    players = _player_order(summaries)
    if not players:
        return
    rews = {p: [s["final_rewards"].get(p, 0) for s in summaries] for p in players}

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(players)), 5))
    rng = np.random.default_rng(42)
    bp  = ax.boxplot(
        [rews[p] for p in players], positions=range(len(players)), widths=0.45,
        patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3,
    )
    for i, (patch, p) in enumerate(zip(bp["boxes"], players)):
        col = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, p in enumerate(players):
        col = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        jit = rng.uniform(-0.15, 0.15, len(rews[p]))
        ax.scatter(np.full(len(rews[p]), i) + jit, rews[p],
                   color=col, alpha=0.45, s=14, zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels(players, rotation=20, ha="right")
    _style(ax, f"Final Reward by Player Position (n={len(summaries)})",
           "Player", "Final reward")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 3: cumulative average reward (stability) ─────────────────────────────

def plot_reward_over_episodes(summaries: list, out: str):
    players = _player_order(summaries)
    if not players:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ep_idx  = np.arange(1, len(summaries) + 1)
    for i, p in enumerate(players):
        col    = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        vals   = np.array([s["final_rewards"].get(p, 0) for s in summaries])
        cumavg = np.cumsum(vals) / ep_idx
        ax.plot(ep_idx, cumavg, color=col, linewidth=1.5, label=p)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    _style(ax, "Cumulative Average Reward per Player", "Episode", "Cumulative mean reward")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 4: fraction of episodes with positive reward per player ──────────────

def plot_survival_rate(summaries: list, out: str):
    """Proxy for survival: players who end with reward > 0 were on the winning team."""
    players = _player_order(summaries)
    if not players:
        return
    total  = len(summaries)
    rates  = [sum(1 for s in summaries if s["final_rewards"].get(p, 0) > 0) / total * 100
              for p in players]
    uniform = 100.0  # all players on winning team get positive reward → not a uniform baseline

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(players)), 5))
    x = np.arange(len(players))
    ax.bar(x, rates, color=[PLAYER_COLORS[i % len(PLAYER_COLORS)] for i in range(len(players))],
           zorder=3)
    for i, r in enumerate(rates):
        ax.text(i, r + 1.5, f"{r:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(players, rotation=20, ha="right")
    ax.set_ylim(0, 115)
    _style(ax, "% Episodes Ending on Winning Team by Player Position",
           "Player", "% episodes with positive reward")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/mafia")
    ap.add_argument("--out",     default="plots/mafia")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summaries = load_summaries(args.log_dir)
    if not summaries:
        print("No summaries found. Run episodes first."); return

    players = _player_order(summaries)
    print(f"Loaded {len(summaries)} episodes, {len(players)} players")
    for p in players:
        mean_r = np.mean([s["final_rewards"].get(p, 0) for s in summaries])
        pos    = sum(1 for s in summaries if s["final_rewards"].get(p, 0) > 0)
        print(f"  {p}: mean reward {mean_r:.3f}, positive in {pos}/{len(summaries)} eps")

    plot_win_rates(summaries,          os.path.join(args.out, "win_rates.png"))
    plot_reward_distribution(summaries, os.path.join(args.out, "reward_distribution.png"))
    plot_reward_over_episodes(summaries, os.path.join(args.out, "reward_over_episodes.png"))
    plot_survival_rate(summaries,       os.path.join(args.out, "survival_rate.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
