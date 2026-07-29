"""Plot SecretMafia experiment results.

Usage:
    python scripts/textarena/plot_mafia_results.py [--log-dir logs/mafia] [--out plots/mafia]

Reads episode_*.summary.json and episode_*.jsonl from logs/mafia/mafia_noop_8p/ and produces:
    win_rates.png             — mafia vs civilian win rate
    reward_distribution.png   — per-player reward spread
    survival_rate.png         — how often each player position survives to end
    reward_over_episodes.png  — cumulative average reward (stability / convergence check)
    elimination_timeline.png  — mean elimination order / turn per player seat
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


def load_episode_logs(log_dir: str) -> list[list[dict]]:
    """Load JSONL step logs — one list-of-steps per episode."""
    episodes = []
    for path in sorted(glob.glob(os.path.join(log_dir, "**", "episode_*.jsonl"),
                                 recursive=True)):
        steps = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        steps.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            continue
        if steps:
            episodes.append(steps)
    return episodes


def _extract_eliminations(steps: list[dict]) -> list[dict]:
    """Return [{player, turn, phase}] in elimination order, best-effort."""
    for step in reversed(steps):
        info = step.get("info", {}) or {}
        close = info.get("close_info", {}) or {}
        if not isinstance(close, dict):
            continue
        for key in ("elimination_order", "eliminations", "dead_players",
                    "eliminated_players", "kill_order", "deaths"):
            val = close.get(key)
            if isinstance(val, list) and val:
                return [{"player": str(p), "turn": None, "phase": "unknown"}
                        for p in val]

    elims, seen = [], set()
    for step in steps:
        info = step.get("info", {}) or {}
        if not isinstance(info, dict):
            continue
        for key in ("eliminated", "killed", "voted_out", "night_kill",
                    "elimination", "lynched"):
            val = info.get(key)
            if val is not None and str(val) not in seen:
                seen.add(str(val))
                elims.append({
                    "player": str(val),
                    "turn": step.get("turn"),
                    "phase": info.get("phase", "unknown"),
                })
    return elims


def _close_info_keys(episode_logs: list[list[dict]]) -> set[str]:
    keys: set[str] = set()
    for steps in episode_logs:
        for step in reversed(steps):
            info = step.get("info", {}) or {}
            close = info.get("close_info", {}) or {}
            if isinstance(close, dict):
                keys.update(close.keys())
            if close:
                break
    return keys


def _player_order(summaries: list[dict]) -> list[str]:
    if not summaries:
        return []
    pids = list(summaries[0]["final_rewards"].keys())
    try:
        return sorted(pids, key=lambda p: int(re.search(r"\d+", p).group()))
    except (AttributeError, TypeError):
        return sorted(pids)


def _winner(summary: dict) -> str:
    info = summary.get("final_info", {})
    close = info.get("close_info", info)
    if isinstance(close, dict):
        w = close.get("winner", close.get("winning_team", ""))
        if w:
            return str(w).lower()
    return "unknown"


# ── plot 1: win rates ─────────────────────────────────────────────────────────

def plot_win_rates(summaries: list, out: str):
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
        ax.bar(x, values, color=colors, zorder=3)
        for i, v in enumerate(values):
            ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, max(values) * 1.3 + 10)
    else:
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


# ── plot 3: cumulative average reward ─────────────────────────────────────────

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


# ── plot 4: survival rate ─────────────────────────────────────────────────────

def plot_survival_rate(summaries: list, out: str):
    players = _player_order(summaries)
    if not players:
        return
    total  = len(summaries)
    rates  = [sum(1 for s in summaries if s["final_rewards"].get(p, 0) > 0) / total * 100
              for p in players]

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


def plot_elimination_timeline(episode_logs: list[list[dict]],
                              summaries: list[dict], out: str):
    players = _player_order(summaries)
    if not players:
        return

    n_players = len(players)
    survived_rank = n_players + 1
    elim_orders: dict[str, list[int]] = defaultdict(list)
    elim_turns:  dict[str, list[int]] = defaultdict(list)
    parsed_any = False

    for steps in episode_logs:
        elims = _extract_eliminations(steps)
        if not elims:
            continue
        parsed_any = True
        elim_set = {e["player"] for e in elims}
        for rank, e in enumerate(elims, start=1):
            p = e["player"]
            elim_orders[p].append(rank)
            if e["turn"] is not None:
                elim_turns[p].append(e["turn"])
        for p in players:
            if p not in elim_set:
                elim_orders[p].append(survived_rank)

    if not parsed_any:
        keys = _close_info_keys(episode_logs)
        print(f"\n[elimination_timeline] No elimination events found in logs.")
        print(f"  close_info keys observed: {sorted(keys) or '(none)'}")
        print(f"  Skipping elimination_timeline.png\n")
        return

    colors = [PLAYER_COLORS[i % len(PLAYER_COLORS)] for i in range(n_players)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Elimination Timeline — SecretMafia (n={len(episode_logs)} episodes)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    means = [np.mean(elim_orders.get(p, [survived_rank])) for p in players]
    sems  = [np.std(elim_orders.get(p, [survived_rank])) /
             max(1, len(elim_orders.get(p, []))) ** 0.5 for p in players]
    x = np.arange(n_players)
    ax.bar(x, means, color=colors, zorder=3, yerr=sems, capsize=5,
           error_kw={"linewidth": 1.2})
    ax.axhline(survived_rank, color="#AAAAAA", linewidth=1, linestyle="--",
               label=f"Survived (rank {survived_rank})")
    for i, m in enumerate(means):
        ax.text(i, m + 0.15, f"{m:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(players, rotation=20, ha="right")
    ax.set_ylabel("Mean elimination order (lower = killed earlier)")
    ax.set_ylim(0, survived_rank * 1.25)
    ax.set_title("Mean Elimination Order by Seat", fontsize=11, fontweight="bold", pad=6)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    data = [elim_orders.get(p, [survived_rank]) for p in players]
    rng = np.random.default_rng(42)
    bp = ax.boxplot(data, positions=range(n_players), widths=0.4,
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=2), zorder=3)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, (d, col) in enumerate(zip(data, colors)):
        jit = rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(np.full(len(d), i) + jit, d,
                   color=col, alpha=0.5, s=14, zorder=4)
    ax.axhline(survived_rank, color="#AAAAAA", linewidth=1, linestyle="--")
    ax.set_xticks(range(n_players))
    ax.set_xticklabels(players, rotation=20, ha="right")
    ax.set_ylabel("Elimination order")
    ax.set_ylim(0, survived_rank * 1.2)
    ax.set_title("Elimination Order Distribution by Seat", fontsize=11, fontweight="bold", pad=6)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if elim_turns:
        note_lines = []
        for p in players:
            turns = elim_turns.get(p)
            if turns:
                note_lines.append(f"{p}: mean turn {np.mean(turns):.1f}")
        if note_lines:
            fig.text(0.5, -0.02, "Mean elimination turn: " + "  |  ".join(note_lines),
                     ha="center", fontsize=8, color="#666666")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
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

    episode_logs = load_episode_logs(args.log_dir)
    print(f"Loaded {len(episode_logs)} JSONL episode log(s) for elimination analysis.")

    plot_win_rates(summaries,          os.path.join(args.out, "win_rates.png"))
    plot_reward_distribution(summaries, os.path.join(args.out, "reward_distribution.png"))
    plot_reward_over_episodes(summaries, os.path.join(args.out, "reward_over_episodes.png"))
    plot_survival_rate(summaries,       os.path.join(args.out, "survival_rate.png"))
    plot_elimination_timeline(episode_logs, summaries,
                              os.path.join(args.out, "elimination_timeline.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
