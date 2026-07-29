"""Plot Debate experiment results.

Usage:
    python scripts/debate/plot_debate_results.py [--log-dir logs/debate] [--out plots/debate]

Reads episode_*.summary.json from logs/debate/debate_noop_2p/ and produces:
    win_rate.png             — which player position wins more often
    reward_distribution.png — reward spread per player
    reward_over_episodes.png — cumulative average (position bias / stability check)
    overview.png             — 2-panel summary

In Debate-v0 one player argues FOR, the other AGAINST. The game is zero-sum:
the winner gets positive reward, the loser gets negative (or zero). Position
bias (player_0 wins more than player_1) would indicate a structural FOR/AGAINST
advantage rather than persuasion quality.
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

FOR_COLOR     = "#4C9BE8"   # player_0 / FOR side
AGAINST_COLOR = "#E8604C"   # player_1 / AGAINST side
GRID_COLOR    = "#DDDDDD"


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


def _side_label(i: int, player: str) -> str:
    labels = ["Affirmative / FOR (player_0)", "Non-Affirmative / AGAINST (player_1)"]
    return labels[i] if i < len(labels) else player


# ── plot 1: win rate per position ─────────────────────────────────────────────

def plot_win_rate(summaries: list, out: str):
    players = _player_order(summaries)
    total   = len(summaries)
    colors  = [FOR_COLOR, AGAINST_COLOR] + ["#4CAF7D"] * max(0, len(players) - 2)

    # A player "wins" if their reward is strictly greater than the opponent's
    wins = []
    for p in players:
        w = sum(
            1 for s in summaries
            if s["final_rewards"].get(p, 0) > max(
                (s["final_rewards"].get(q, 0) for q in players if q != p),
                default=0
            )
        )
        wins.append(w / total * 100)

    ci = [1.96 * (w/100 * (1 - w/100) / total) ** 0.5 * 100 for w in wins]

    x = np.arange(len(players))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(x, wins, color=colors[:len(players)], zorder=3,
           yerr=ci, capsize=6, error_kw={"linewidth": 1.2})
    ax.axhline(50, color="#AAAAAA", linewidth=1.2, linestyle="--", label="50% (random)")
    for i, (w, c) in enumerate(zip(wins, ci)):
        ax.text(i, w + c + 2, f"{w:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([_side_label(i, p) for i, p in enumerate(players)])
    ax.set_ylim(0, max(wins) * 1.35 + 10)
    _style(ax, f"Win Rate by Debate Position (n={total})", "Position", "Win rate (%)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 2: reward distribution ───────────────────────────────────────────────

def plot_reward_distribution(summaries: list, out: str):
    players = _player_order(summaries)
    colors  = [FOR_COLOR, AGAINST_COLOR] + ["#4CAF7D"] * max(0, len(players) - 2)
    rews    = {p: [s["final_rewards"].get(p, 0) for s in summaries] for p in players}

    fig, ax = plt.subplots(figsize=(6, 5))
    rng = np.random.default_rng(42)
    bp  = ax.boxplot(
        [rews[p] for p in players], positions=range(len(players)), widths=0.4,
        patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3,
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, (p, col) in enumerate(zip(players, colors)):
        jit = rng.uniform(-0.12, 0.12, len(rews[p]))
        ax.scatter(np.full(len(rews[p]), i) + jit, rews[p],
                   color=col, alpha=0.5, s=16, zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels([_side_label(i, p) for i, p in enumerate(players)])
    _style(ax, f"Reward Distribution by Position (n={len(summaries)})",
           "Position", "Final reward")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 3: cumulative average (bias / stability) ─────────────────────────────

def plot_reward_over_episodes(summaries: list, out: str):
    players = _player_order(summaries)
    colors  = [FOR_COLOR, AGAINST_COLOR] + ["#4CAF7D"] * max(0, len(players) - 2)
    fig, ax = plt.subplots(figsize=(9, 5))
    ep_idx  = np.arange(1, len(summaries) + 1)
    for p, col in zip(players, colors):
        vals   = np.array([s["final_rewards"].get(p, 0) for s in summaries])
        cumavg = np.cumsum(vals) / ep_idx
        ax.plot(ep_idx, cumavg, color=col, linewidth=2,
                label=_side_label(players.index(p), p))
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--", label="Zero")
    _style(ax, "Cumulative Average Reward — Position Bias Check",
           "Episode", "Cumulative mean reward")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 4: overview ──────────────────────────────────────────────────────────

def plot_overview(summaries: list, out: str):
    players = _player_order(summaries)
    colors  = [FOR_COLOR, AGAINST_COLOR] + ["#4CAF7D"] * max(0, len(players) - 2)
    total   = len(summaries)
    rews    = {p: [s["final_rewards"].get(p, 0) for s in summaries] for p in players}

    wins = []
    for p in players:
        w = sum(1 for s in summaries
                if s["final_rewards"].get(p, 0) > max(
                    (s["final_rewards"].get(q, 0) for q in players if q != p), default=0))
        wins.append(w / total * 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    topics = list(dict.fromkeys(t for t in (_extract_topic(s) for s in summaries) if t))
    topic_str = f": "{topics[0]}"" if len(topics) == 1 else ""
    fig.suptitle(f"Debate Baseline Results (n={total}){topic_str}", fontsize=13, fontweight="bold")

    ax = axes[0]
    x  = np.arange(len(players))
    ax.bar(x, wins, color=colors[:len(players)], zorder=3)
    ax.axhline(50, color="#AAAAAA", linewidth=1.2, linestyle="--", label="50%")
    for i, w in enumerate(wins):
        ax.text(i, w + 1.5, f"{w:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([_side_label(i, p) for i, p in enumerate(players)])
    ax.set_ylim(0, max(wins) * 1.35 + 10)
    _style(ax, "Win Rate by Position", "Position", "Win rate (%)")
    ax.legend(frameon=False, fontsize=9)

    ax  = axes[1]
    rng = np.random.default_rng(0)
    bp  = ax.boxplot(
        [rews[p] for p in players], positions=range(len(players)), widths=0.4,
        patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3,
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, (p, col) in enumerate(zip(players, colors)):
        jit = rng.uniform(-0.1, 0.1, len(rews[p]))
        ax.scatter(np.full(len(rews[p]), i) + jit, rews[p],
                   color=col, alpha=0.45, s=12, zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels([_side_label(i, p) for i, p in enumerate(players)])
    _style(ax, "Reward Distribution", "Position", "Final reward")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def _extract_topic(summary: dict) -> str | None:
    """Extract the debate motion/topic from close_info."""
    ci = (summary.get("final_info") or {}).get("close_info") or {}
    if not isinstance(ci, dict):
        return None
    for key in ("topic", "motion", "debate_topic", "statement", "proposition",
                "resolution", "subject", "claim"):
        val = ci.get(key)
        if val:
            return str(val)
    return None


def _print_summary(summaries: list[dict]) -> None:
    players = _player_order(summaries)
    total   = len(summaries)

    position_labels = {
        "player_0": "FOR  (argues IN FAVOUR of the motion)",
        "player_1": "AGAINST (argues AGAINST the motion)",
    }

    # ── role framing ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Debate baseline — {total} episode(s)")
    print(f"{'─'*60}")
    print(f"  Player roles:")
    for i, p in enumerate(players):
        label = position_labels.get(p, f"position {i}")
        print(f"    {p}  →  {label}")

    # ── debate topics ─────────────────────────────────────────────────────────
    topics = [_extract_topic(s) for s in summaries]
    topics_found = [(i, t) for i, t in enumerate(topics) if t]
    if topics_found:
        unique_topics = list(dict.fromkeys(t for _, t in topics_found))
        print(f"\n  Debate motions ({len(unique_topics)} unique across {total} episodes):")
        for t in unique_topics[:10]:
            eps = [i for i, topic in topics_found if topic == t]
            ep_str = ", ".join(str(e) for e in eps[:5])
            if len(eps) > 5:
                ep_str += f" … ({len(eps)} total)"
            print(f"    • {t}  [ep {ep_str}]")
        if len(unique_topics) > 10:
            print(f"    … and {len(unique_topics) - 10} more motions")
    else:
        print(f"\n  Debate motions: not found in close_info")

    # ── win/reward stats ──────────────────────────────────────────────────────
    win_counts = {p: 0 for p in players}
    draw_count = 0
    for s in summaries:
        rews    = s["final_rewards"]
        best    = max(rews.get(p, 0) for p in players)
        winners = [p for p in players if rews.get(p, 0) == best]
        if len(winners) > 1:
            draw_count += 1
        else:
            win_counts[winners[0]] += 1

    print(f"\n  {'Player':<12}  {'Position':<38}  {'Wins':>5}  {'Win%':>6}  "
          f"{'Mean rew':>9}  {'Std':>7}  {'Median':>7}")
    print(f"  {'─'*12}  {'─'*38}  {'─'*5}  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*7}")
    for i, p in enumerate(players):
        rews  = [s["final_rewards"].get(p, 0) for s in summaries]
        label = position_labels.get(p, f"position {i}")
        wins  = win_counts[p]
        print(f"  {p:<12}  {label:<38}  {wins:>5}  {wins/total*100:>5.1f}%  "
              f"{np.mean(rews):>9.3f}  {np.std(rews):>7.3f}  {float(np.median(rews)):>7.3f}")
    print(f"  {'─'*60}")
    print(f"  Draws: {draw_count}/{total} ({draw_count/total*100:.1f}%)")

    # ── remaining close_info keys ─────────────────────────────────────────────
    close_keys: dict = {}
    for s in summaries:
        ci = (s.get("final_info") or {}).get("close_info") or {}
        for k, v in ci.items():
            if k not in ("topic", "motion", "debate_topic", "statement",
                         "proposition", "resolution", "subject", "claim"):
                close_keys.setdefault(k, []).append(v)
    if close_keys:
        print(f"\n  Other close_info fields:")
        for k, vals in sorted(close_keys.items()):
            unique = list(dict.fromkeys(str(v) for v in vals))
            sample = unique[:5]
            suffix = f" … ({len(unique)} unique)" if len(unique) > 5 else ""
            print(f"    {k}: {', '.join(sample)}{suffix}")
    print(f"{'─'*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/debate")
    ap.add_argument("--out",     default="plots/debate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summaries = load_summaries(args.log_dir)
    if not summaries:
        print("No summaries found. Run episodes first."); return

    _print_summary(summaries)

    plot_win_rate(summaries,           os.path.join(args.out, "win_rate.png"))
    plot_reward_distribution(summaries, os.path.join(args.out, "reward_distribution.png"))
    plot_reward_over_episodes(summaries, os.path.join(args.out, "reward_over_episodes.png"))
    plot_overview(summaries,            os.path.join(args.out, "overview.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
