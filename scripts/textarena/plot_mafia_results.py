"""Plot SecretMafia experiment results — role-aware.

Usage:
    python scripts/textarena/plot_mafia_results.py [--log-dir logs/mafia] [--out plots/mafia]

Reads episode_*.summary.json and episode_*.jsonl from logs/mafia/mafia_noop_8p/.

Because SecretMafia randomly assigns roles (Mafia, Doctor, Detective, Civilian) to
player seats each episode, seat-based win rates are meaningless. All plots group
observations by ROLE extracted from close_info. When role data is unavailable the
script prints which close_info keys ARE present and falls back to seat-based plots
so the output is always useful.

Plots produced:
    role_win_rate.png          — win % per role (Mafia vs Town/Civilian etc.)
    role_reward_distribution.png — reward spread per role across episodes
    role_frequency.png         — how often each role appears (sanity check)
    elimination_timeline.png   — mean elimination order per role
    seat_reward.png            — fallback seat-based reward (always emitted)
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

GRID_COLOR = "#DDDDDD"

# Colour palette per role — extended as needed by _role_color()
ROLE_COLORS: dict[str, str] = {
    "mafia":      "#E8604C",
    "godfather":  "#C0392B",
    "doctor":     "#4CAF7D",
    "detective":  "#4C9BE8",
    "civilian":   "#95A5A6",
    "villager":   "#95A5A6",
    "town":       "#95A5A6",
    "sheriff":    "#F5A623",
    "jester":     "#9B59B6",
}
_FALLBACK_COLORS = [
    "#4C9BE8", "#E8604C", "#4CAF7D", "#F5A623",
    "#9B59B6", "#1ABC9C", "#E74C3C", "#95A5A6",
]


def _role_color(role: str, idx: int = 0) -> str:
    return ROLE_COLORS.get(role.lower(), _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── data loading ──────────────────────────────────────────────────────────────

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


def _player_order(summaries: list[dict]) -> list[str]:
    if not summaries:
        return []
    pids = list(summaries[0]["final_rewards"].keys())
    try:
        return sorted(pids, key=lambda p: int(re.search(r"\d+", p).group()))
    except (AttributeError, TypeError):
        return sorted(pids)


# ── role extraction ───────────────────────────────────────────────────────────

def _extract_roles(summary: dict) -> dict[str, str] | None:
    """Return {player_id: role_name} from summary, or None if not found.

    Checks the top-level player_roles field first (written by the orchestrator
    since it now persists roles for every game), then falls back to scanning
    close_info for older logs that pre-date that field.
    """
    top = summary.get("player_roles")
    if isinstance(top, dict) and top:
        return {str(k): str(v).lower() for k, v in top.items()}

    ci = (summary.get("final_info") or {}).get("close_info") or {}
    if not isinstance(ci, dict):
        return None
    for key in ("roles", "player_roles", "role_assignments", "role_map",
                "assignments", "player_role"):
        val = ci.get(key)
        if isinstance(val, dict) and val:
            return {str(k): str(v).lower() for k, v in val.items()}
    return None


def _extract_winner_team(summary: dict) -> str | None:
    """Return winning team/role string from close_info."""
    ci = (summary.get("final_info") or {}).get("close_info") or {}
    if not isinstance(ci, dict):
        return None
    for key in ("winner", "winning_team", "winning_role", "team_winner"):
        val = ci.get(key)
        if val:
            return str(val).lower()
    return None


def _close_info_keys(summaries: list[dict]) -> set[str]:
    keys: set[str] = set()
    for s in summaries:
        ci = (s.get("final_info") or {}).get("close_info") or {}
        if isinstance(ci, dict):
            keys.update(ci.keys())
    return keys


# ── aggregate role-level stats ────────────────────────────────────────────────

def _role_rewards(summaries: list[dict]) -> dict[str, list[float]]:
    """Map role name → list of rewards across all (episode, player) pairs."""
    result: dict[str, list[float]] = defaultdict(list)
    for s in summaries:
        roles = _extract_roles(s)
        if not roles:
            continue
        rews = s["final_rewards"]
        for pid, role in roles.items():
            if pid in rews:
                result[role].append(rews[pid])
    return dict(result)


def _role_wins(summaries: list[dict], role_rews: dict[str, list[float]]) -> dict[str, int]:
    """Count episodes where each team/role won."""
    wins: dict[str, int] = defaultdict(int)
    for s in summaries:
        winner = _extract_winner_team(s)
        if winner:
            wins[winner] += 1
            continue
        # Fallback: winning role = role(s) with max reward in this episode
        roles = _extract_roles(s)
        if not roles:
            continue
        rews = s["final_rewards"]
        best = max((rews.get(pid, float("-inf")) for pid in roles), default=None)
        if best is None:
            continue
        for pid, role in roles.items():
            if rews.get(pid) == best:
                wins[role] += 1
                break
    return dict(wins)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_role_win_rate(summaries: list, out: str):
    wins = _role_wins(summaries, _role_rewards(summaries))
    if not wins:
        print(f"[role_win_rate] No role data — skipping {out}")
        return

    total = len(summaries)
    roles = sorted(wins)
    colors = [_role_color(r, i) for i, r in enumerate(roles)]
    values = [wins[r] / total * 100 for r in roles]

    fig, ax = plt.subplots(figsize=(max(6, len(roles) * 1.5), 5))
    x = np.arange(len(roles))
    ax.bar(x, values, color=colors, zorder=3)
    for i, v in enumerate(values):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in roles])
    ax.set_ylim(0, max(values) * 1.35 + 10)
    _style(ax, f"SecretMafia — Win Rate by Role (n={total} episodes)",
           "Role", "% episodes won")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_role_reward_distribution(summaries: list, out: str):
    role_rews = _role_rewards(summaries)
    if not role_rews:
        print(f"[role_reward_distribution] No role data — skipping {out}")
        return

    roles = sorted(role_rews)
    colors = [_role_color(r, i) for i, r in enumerate(roles)]
    data = [role_rews[r] for r in roles]

    fig, ax = plt.subplots(figsize=(max(7, len(roles) * 1.8), 5))
    rng = np.random.default_rng(42)
    bp = ax.boxplot(data, positions=range(len(roles)), widths=0.45,
                    patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, (d, col) in enumerate(zip(data, colors)):
        jit = rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(np.full(len(d), i) + jit, d, color=col, alpha=0.45, s=14, zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels([f"{r.capitalize()}\n(n={len(role_rews[r])})" for r in roles])
    _style(ax, f"SecretMafia — Reward Distribution by Role",
           "Role", "Final reward")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_role_frequency(summaries: list, out: str):
    """Sanity check: how many times does each role appear across all episodes?"""
    freq: dict[str, int] = defaultdict(int)
    for s in summaries:
        roles = _extract_roles(s)
        if roles:
            for role in roles.values():
                freq[role] += 1
    if not freq:
        print(f"[role_frequency] No role data — skipping {out}")
        return

    roles = sorted(freq)
    colors = [_role_color(r, i) for i, r in enumerate(roles)]
    counts = [freq[r] for r in roles]
    total_assignments = sum(counts)

    fig, ax = plt.subplots(figsize=(max(6, len(roles) * 1.5), 5))
    x = np.arange(len(roles))
    ax.bar(x, counts, color=colors, zorder=3)
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, f"{c}\n({c/total_assignments*100:.0f}%)",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in roles])
    _style(ax, f"SecretMafia — Role Frequency (n={len(summaries)} episodes)",
           "Role", "# player-episode assignments")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_elimination_by_role(episode_logs: list[list[dict]],
                             summaries: list[dict], out: str):
    """Elimination order grouped by role instead of seat."""
    from collections import defaultdict as _dd

    def _extract_elims(steps):
        for step in reversed(steps):
            ci = (step.get("info") or {}).get("close_info") or {}
            for key in ("elimination_order", "eliminations", "dead_players",
                        "eliminated_players", "kill_order", "deaths"):
                val = ci.get(key)
                if isinstance(val, list) and val:
                    return [str(p) for p in val]
        elims, seen = [], set()
        for step in steps:
            info = step.get("info") or {}
            for key in ("eliminated", "killed", "voted_out", "night_kill", "lynched"):
                val = info.get(key)
                if val is not None and str(val) not in seen:
                    seen.add(str(val)); elims.append(str(val))
        return elims

    # Pair each episode log with its summary to get roles
    role_elim_orders: dict[str, list[int]] = _dd(list)
    parsed = 0
    for steps, summary in zip(episode_logs, summaries):
        roles = _extract_roles(summary)
        if not roles:
            continue
        elims = _extract_elims(steps)
        if not elims:
            continue
        parsed += 1
        n_players = len(roles)
        survived_rank = n_players + 1
        elim_set = set(elims)
        for rank, pid in enumerate(elims, start=1):
            role = roles.get(pid, "unknown")
            role_elim_orders[role].append(rank)
        for pid, role in roles.items():
            if pid not in elim_set:
                role_elim_orders[role].append(survived_rank)

    if not role_elim_orders:
        print(f"[elimination_by_role] No paired role+elimination data — skipping {out}")
        return

    roles = sorted(role_elim_orders)
    colors = [_role_color(r, i) for i, r in enumerate(roles)]
    data = [role_elim_orders[r] for r in roles]
    survived_rank = max(max(d) for d in data if d)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"SecretMafia — Elimination Timeline by Role (n={parsed} episodes)",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    means = [np.mean(d) for d in data]
    sems  = [np.std(d) / max(1, len(d)) ** 0.5 for d in data]
    x = np.arange(len(roles))
    ax.bar(x, means, color=colors, zorder=3, yerr=sems, capsize=5,
           error_kw={"linewidth": 1.2})
    ax.axhline(survived_rank, color="#AAAAAA", linewidth=1, linestyle="--",
               label=f"Survived (rank {survived_rank})")
    for i, m in enumerate(means):
        ax.text(i, m + 0.15, f"{m:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in roles])
    ax.set_ylim(0, survived_rank * 1.3)
    ax.set_ylabel("Mean elimination order (lower = killed earlier)")
    ax.set_title("Mean Elimination Order by Role", fontsize=11, fontweight="bold", pad=6)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    rng = np.random.default_rng(42)
    bp = ax.boxplot(data, positions=range(len(roles)), widths=0.4,
                    patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, (d, col) in enumerate(zip(data, colors)):
        jit = rng.uniform(-0.13, 0.13, len(d))
        ax.scatter(np.full(len(d), i) + jit, d, color=col, alpha=0.5, s=14, zorder=4)
    ax.axhline(survived_rank, color="#AAAAAA", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels([f"{r.capitalize()}\n(n={len(role_elim_orders[r])})"
                        for r in roles])
    ax.set_ylabel("Elimination order")
    ax.set_ylim(0, survived_rank * 1.2)
    ax.set_title("Elimination Distribution by Role", fontsize=11, fontweight="bold", pad=6)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def plot_seat_reward(summaries: list, out: str):
    """Seat-based reward — always emitted as a sanity/comparison plot."""
    players = _player_order(summaries)
    if not players:
        return
    rews = {p: [s["final_rewards"].get(p, 0) for s in summaries] for p in players}

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(players)), 5))
    rng = np.random.default_rng(42)
    bp = ax.boxplot(
        [rews[p] for p in players], positions=range(len(players)), widths=0.45,
        patch_artist=True, medianprops=dict(color="black", linewidth=2), zorder=3,
    )
    for i, (patch, p) in enumerate(zip(bp["boxes"], players)):
        col = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
        patch.set_facecolor(col + "55"); patch.set_edgecolor(col)
    for i, p in enumerate(players):
        col = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
        jit = rng.uniform(-0.15, 0.15, len(rews[p]))
        ax.scatter(np.full(len(rews[p]), i) + jit, rews[p],
                   color=col, alpha=0.45, s=14, zorder=4)
    ax.axhline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels(players, rotation=20, ha="right")
    _style(ax, f"SecretMafia — Reward by Seat (role-blind; n={len(summaries)})",
           "Player seat", "Final reward")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── console summary ───────────────────────────────────────────────────────────

def _print_summary(summaries: list[dict]) -> None:
    total = len(summaries)
    role_rews = _role_rewards(summaries)
    wins = _role_wins(summaries, role_rews)

    print(f"\n{'─'*64}")
    print(f"  SecretMafia — {total} episode(s)")
    print(f"{'─'*64}")

    if role_rews:
        print(f"  Role-level statistics:")
        print(f"  {'Role':<14}  {'# obs':>6}  {'Wins':>5}  {'Win%':>6}  "
              f"{'Mean rew':>9}  {'Std':>7}  {'Median':>7}")
        print(f"  {'─'*14}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*7}")
        for role in sorted(role_rews):
            rews = role_rews[role]
            w = wins.get(role, 0)
            print(f"  {role.capitalize():<14}  {len(rews):>6}  {w:>5}  "
                  f"{w/total*100:>5.1f}%  "
                  f"{np.mean(rews):>9.3f}  {np.std(rews):>7.3f}  "
                  f"{float(np.median(rews)):>7.3f}")
    else:
        keys = _close_info_keys(summaries)
        print(f"  Role data not found in close_info.")
        print(f"  Available close_info keys: {sorted(keys) or '(none)'}")
        print(f"  → Falling back to seat-based plots only.")
        players = _player_order(summaries)
        print(f"\n  Seat-level (role-blind):")
        for p in players:
            rews = [s["final_rewards"].get(p, 0) for s in summaries]
            print(f"    {p}: mean {np.mean(rews):.3f}  std {np.std(rews):.3f}")

    # Extra close_info fields
    extra: dict = {}
    for s in summaries:
        ci = (s.get("final_info") or {}).get("close_info") or {}
        for k, v in ci.items():
            if k not in ("roles", "player_roles", "role_assignments", "role_map",
                         "assignments", "player_role", "winner", "winning_team",
                         "winning_role", "team_winner"):
                extra.setdefault(k, []).append(v)
    if extra:
        print(f"\n  Other close_info fields:")
        for k, vals in sorted(extra.items()):
            unique = list(dict.fromkeys(str(v) for v in vals))
            sample = unique[:5]
            suffix = f" … ({len(unique)} unique)" if len(unique) > 5 else ""
            print(f"    {k}: {', '.join(sample)}{suffix}")
    print(f"{'─'*64}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/mafia")
    ap.add_argument("--out",     default="plots/mafia")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summaries = load_summaries(args.log_dir)
    if not summaries:
        print("No summaries found. Run episodes first."); return

    episode_logs = load_episode_logs(args.log_dir)
    print(f"Loaded {len(summaries)} episode summaries, {len(episode_logs)} JSONL logs.")

    _print_summary(summaries)

    plot_role_win_rate(summaries,
                       os.path.join(args.out, "role_win_rate.png"))
    plot_role_reward_distribution(summaries,
                                  os.path.join(args.out, "role_reward_distribution.png"))
    plot_role_frequency(summaries,
                        os.path.join(args.out, "role_frequency.png"))
    plot_elimination_by_role(episode_logs, summaries,
                             os.path.join(args.out, "elimination_timeline.png"))
    plot_seat_reward(summaries,
                     os.path.join(args.out, "seat_reward.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
