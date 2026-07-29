"""Plot Public Goods Game experiment results.

Usage:
    python scripts/textarena/plot_public_goods_results.py \
        [--log-dir logs/public_goods] [--out plots/public_goods]

In PublicGoodsGame-v0 each agent simultaneously decides how much of their
endowment to contribute to a shared pool, which is then multiplied and split
equally among all players. The tension between individual free-riding and
collective maximisation drives the strategic dynamics. The plots reveal which
seat contributes most, how rewards distribute across positions, and whether
cumulative performance stabilises as agents calibrate cooperation over episodes.

Note: the key metric is mean contribution (cooperation rate), which can be
inferred from rewards if close_info exposes contribution amounts; higher mean
contributions indicate more cooperative behaviour.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GAME_NAME = "Public Goods Game"
PLAYER_LABELS = {
    "player_0": "Agent 0",
    "player_1": "Agent 1",
    "player_2": "Agent 2",
    "player_3": "Agent 3",
}
PLAYER_COLORS = [
    "#4C9BE8", "#E8604C", "#4CAF7D", "#F5A623",
    "#9B59B6", "#1ABC9C", "#E74C3C", "#95A5A6",
    "#34495E", "#F39C12",
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


def _player_label(p: str) -> str:
    return PLAYER_LABELS.get(p, p)


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


def plot_win_rate(summaries: list, out: str):
    players = _player_order(summaries)
    total = len(summaries)
    colors = [PLAYER_COLORS[i % len(PLAYER_COLORS)] for i in range(len(players))]

    wins = []
    for p in players:
        w = sum(
            1 for s in summaries
            if s["final_rewards"].get(p, 0) > max(
                (s["final_rewards"].get(q, 0) for q in players if q != p), default=0)
        )
        wins.append(w / total * 100)
    ci = [1.96 * (w/100 * (1 - w/100) / total) ** 0.5 * 100 for w in wins]

    x = np.arange(len(players))
    fig, ax = plt.subplots(figsize=(max(6, len(players) * 1.4), 5))
    ax.bar(x, wins, color=colors, zorder=3, yerr=ci, capsize=5,
           error_kw={"linewidth": 1.2})
    ax.axhline(100 / len(players), color="#AAAAAA", linewidth=1.2,
               linestyle="--", label=f"Random ({100/len(players):.0f}%)")
    for i, (w, c) in enumerate(zip(wins, ci)):
        ax.text(i, w + c + 1.5, f"{w:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([_player_label(p) for p in players])
    ax.set_ylim(0, max(wins + [100/len(players)]) * 1.4 + 10)
    _style(ax, f"{GAME_NAME} — Win Rate by Seat (n={total})", "Player", "Win rate (%)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_reward_distribution(summaries: list, out: str):
    players = _player_order(summaries)
    rews = {p: [s["final_rewards"].get(p, 0) for s in summaries] for p in players}
    colors = [PLAYER_COLORS[i % len(PLAYER_COLORS)] for i in range(len(players))]

    fig, ax = plt.subplots(figsize=(max(6, len(players) * 1.5), 5))
    rng = np.random.default_rng(42)
    bp = ax.boxplot(
        [rews[p] for p in players], positions=range(len(players)), widths=0.4,
        patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3,
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, (p, col) in enumerate(zip(players, colors)):
        jit = rng.uniform(-0.13, 0.13, len(rews[p]))
        ax.scatter(np.full(len(rews[p]), i) + jit, rews[p],
                   color=col, alpha=0.45, s=14, zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels([_player_label(p) for p in players])
    _style(ax, f"{GAME_NAME} — Reward Distribution (n={len(summaries)})",
           "Player", "Final reward")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_cumulative_reward(summaries: list, out: str):
    players = _player_order(summaries)
    ep_idx = np.arange(1, len(summaries) + 1)
    colors = [PLAYER_COLORS[i % len(PLAYER_COLORS)] for i in range(len(players))]

    fig, ax = plt.subplots(figsize=(9, 5))
    for p, col in zip(players, colors):
        vals = np.array([s["final_rewards"].get(p, 0) for s in summaries])
        ax.plot(ep_idx, np.cumsum(vals) / ep_idx, color=col, linewidth=2,
                label=_player_label(p))
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    _style(ax, f"{GAME_NAME} — Cumulative Average Reward",
           "Episode", "Cumulative mean reward")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def _print_summary(summaries: list[dict]) -> None:
    players = _player_order(summaries)
    total = len(summaries)
    win_counts = {p: 0 for p in players}
    draw_count = 0
    for s in summaries:
        rews = s["final_rewards"]
        best = max(rews.get(p, 0) for p in players)
        winners = [p for p in players if rews.get(p, 0) == best]
        if len(winners) > 1:
            draw_count += 1
        else:
            win_counts[winners[0]] += 1

    print(f"\n{'─'*60}")
    print(f"  {GAME_NAME} — {total} episode(s)")
    print(f"{'─'*60}")
    print(f"  {'Player':<12}  {'Seat':<12}  {'Wins':>5}  {'Win%':>6}  "
          f"{'Mean rew':>9}  {'Std':>7}  {'Median':>7}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*5}  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*7}")
    for p in players:
        rews = [s["final_rewards"].get(p, 0) for s in summaries]
        wins = win_counts[p]
        print(f"  {p:<12}  {_player_label(p):<12}  {wins:>5}  {wins/total*100:>5.1f}%  "
              f"{np.mean(rews):>9.3f}  {np.std(rews):>7.3f}  {float(np.median(rews)):>7.3f}")
    print(f"  {'─'*60}")
    print(f"  Draws: {draw_count}/{total} ({draw_count/total*100:.1f}%)")

    close_keys: dict = {}
    for s in summaries:
        ci = (s.get("final_info") or {}).get("close_info") or {}
        for k, v in ci.items():
            close_keys.setdefault(k, []).append(v)
    if close_keys:
        print(f"\n  close_info fields:")
        for k, vals in sorted(close_keys.items()):
            unique = list(dict.fromkeys(str(v) for v in vals))
            sample = unique[:5]
            suffix = f" … ({len(unique)} unique)" if len(unique) > 5 else ""
            print(f"    {k}: {', '.join(sample)}{suffix}")
    print(f"{'─'*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/public_goods")
    ap.add_argument("--out",     default="plots/public_goods")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summaries = load_summaries(args.log_dir)
    if not summaries:
        print("No summaries found. Run episodes first."); return

    _print_summary(summaries)
    plot_win_rate(summaries,          os.path.join(args.out, "win_rate.png"))
    plot_reward_distribution(summaries, os.path.join(args.out, "reward_distribution.png"))
    plot_cumulative_reward(summaries, os.path.join(args.out, "cumulative_reward.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
