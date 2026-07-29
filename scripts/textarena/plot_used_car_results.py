"""Plot Used Car Negotiation experiment results.

Usage:
    python scripts/textarena/plot_used_car_results.py \
        [--log-dir logs/used_car] [--out plots/used_car]

In UsedCarNegotiation-v0 a Seller and Buyer negotiate a used car price over multiple
rounds. Each player has a private reservation price (minimum acceptable for Seller,
maximum acceptable for Buyer). A deal is struck when both agree on a price within
the Zone of Possible Agreement; no deal if timeout or incompatible positions.
Key questions: deal rate, who captures more surplus, and position advantage.
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

GAME_NAME = "Used Car Negotiation"
PLAYER_LABELS = {
    "player_0": "Seller",
    "player_1": "Buyer",
}
PLAYER_COLORS = [
    "#E8604C", "#4C9BE8",
    "#4CAF7D", "#F5A623",
    "#9B59B6", "#1ABC9C",
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


def plot_reward_distribution(summaries: list, out: str):
    players = _player_order(summaries)
    rews = {p: [s["final_rewards"].get(p, 0) for s in summaries] for p in players}
    colors = [PLAYER_COLORS[i % len(PLAYER_COLORS)] for i in range(len(players))]

    fig, ax = plt.subplots(figsize=(6, 5))
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
           "Role", "Final reward")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_deal_analysis(summaries: list, out: str):
    players = _player_order(summaries)
    ep_idx = np.arange(1, len(summaries) + 1)
    colors = [PLAYER_COLORS[i % len(PLAYER_COLORS)] for i in range(len(players))]

    joint = np.array([sum(s["final_rewards"].get(p, 0) for p in players)
                      for s in summaries])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    deal_rate = np.cumsum(joint > 0) / ep_idx * 100
    ax.plot(ep_idx, deal_rate, color="#4CAF7D", linewidth=2.5, label="Deal rate")
    ax.set_ylim(0, 105)
    _style(ax, "Cumulative Deal Rate", "Episode", "Deals struck (%)")

    ax = axes[1]
    for p, col in zip(players, colors):
        vals = np.array([s["final_rewards"].get(p, 0) for s in summaries])
        ax.plot(ep_idx, np.cumsum(vals) / ep_idx, color=col, linewidth=2,
                label=_player_label(p))
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    _style(ax, "Cumulative Average Reward by Role",
           "Episode", "Cumulative mean reward")
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"{GAME_NAME}", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def _print_summary(summaries: list[dict]) -> None:
    players = _player_order(summaries)
    total = len(summaries)
    joint_rews = [sum(s["final_rewards"].get(p, 0) for p in players) for s in summaries]
    deal_count = sum(1 for j in joint_rews if j > 0)

    print(f"\n{'─'*60}")
    print(f"  {GAME_NAME} — {total} episode(s)")
    print(f"  Deals struck: {deal_count}/{total} ({deal_count/total*100:.1f}%)")
    print(f"  No-deal rate: {(total-deal_count)/total*100:.1f}% (both get 0)")
    print(f"{'─'*60}")
    print(f"  {'Player':<12}  {'Role':<10}  {'Mean rew':>9}  {'Std':>7}  "
          f"{'Median':>7}  {'% positive':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*10}")
    for p in players:
        rews = [s["final_rewards"].get(p, 0) for s in summaries]
        pos_rate = sum(1 for r in rews if r > 0) / total * 100
        print(f"  {p:<12}  {_player_label(p):<10}  "
              f"{np.mean(rews):>9.3f}  {np.std(rews):>7.3f}  "
              f"{float(np.median(rews)):>7.3f}  {pos_rate:>9.1f}%")

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
    ap.add_argument("--log-dir", default="logs/used_car")
    ap.add_argument("--out",     default="plots/used_car")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summaries = load_summaries(args.log_dir)
    if not summaries:
        print("No summaries found. Run episodes first."); return

    _print_summary(summaries)
    plot_reward_distribution(summaries, os.path.join(args.out, "reward_distribution.png"))
    plot_deal_analysis(summaries,       os.path.join(args.out, "deal_analysis.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
