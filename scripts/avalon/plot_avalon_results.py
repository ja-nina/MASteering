"""Plot Avalon experiment results across player counts.

Usage:
    python scripts/avalon/plot_avalon_results.py [--log-dir logs/avalon] [--out plots/avalon]

Reads episode_*.summary.json from logs/avalon/avalon_noop_{N}p/ for N in 5..10 and produces:
    win_rates.png          — good vs evil win rate by player count
    evil_breakdown.png     — how evil wins (quest failures / rejection / assassination)
    merlin_safety.png      — Merlin assassination rate among games where it could happen
    overview.png           — 3-panel summary figure
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

GOOD_COLOR  = "#4C9BE8"
EVIL_COLOR  = "#E8604C"
QUEST_COLOR = "#E67E22"
REJCT_COLOR = "#95A5A6"
ASSN_COLOR  = "#9B59B6"
GRID_COLOR  = "#DDDDDD"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ci95(n, p):
    """Wilson score 95% CI half-width (returns 0 for n=0)."""
    if n == 0:
        return 0.0
    z = 1.96
    centre = (p + z*z / (2*n)) / (1 + z*z / n)
    half   = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return min(centre - (p - half), half)  # symmetric approx


def load_summaries(log_dir: str) -> dict[int, list[dict]]:
    pattern = re.compile(r"avalon_noop_(\d+)p$")
    data: dict[int, list[dict]] = defaultdict(list)
    for run_dir in sorted(glob.glob(os.path.join(log_dir, "avalon_noop_*p"))):
        m = pattern.search(run_dir)
        if not m:
            continue
        n = int(m.group(1))
        for path in glob.glob(os.path.join(run_dir, "episode_*.summary.json")):
            try:
                with open(path) as f:
                    s = json.load(f)
                info = s.get("final_info", {})
                if info and "winner" in info:
                    data[n].append(info)
            except (json.JSONDecodeError, OSError):
                continue
    return dict(data)


# ── plot 1: win rates ─────────────────────────────────────────────────────────

def plot_win_rates(data: dict, out: str):
    counts = sorted(data)
    good_r, evil_r, ns = [], [], []
    good_ci, evil_ci   = [], []
    for n in counts:
        eps   = data[n]
        total = len(eps)
        good  = sum(1 for e in eps if e.get("winner") == "good")
        p_g   = good / total if total else 0
        p_e   = 1 - p_g
        good_r.append(p_g * 100); evil_r.append(p_e * 100)
        good_ci.append(_ci95(total, p_g) * 100)
        evil_ci.append(_ci95(total, p_e) * 100)
        ns.append(total)

    x = np.arange(len(counts))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, good_r, w, color=GOOD_COLOR, label="Good wins",  zorder=3,
           yerr=good_ci, capsize=4, error_kw={"linewidth": 1.2})
    ax.bar(x + w/2, evil_r, w, color=EVIL_COLOR, label="Evil wins",  zorder=3,
           yerr=evil_ci, capsize=4, error_kw={"linewidth": 1.2})
    ax.axhline(50, color="#AAAAAA", linewidth=1, linestyle="--", zorder=2)
    for i, (g, e, n) in enumerate(zip(good_r, evil_r, ns)):
        ax.text(i - w/2, g + 3, f"{g:.0f}%", ha="center", fontsize=8)
        ax.text(i + w/2, e + 3, f"{e:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}p\n(n={ep})" for n, ep in zip(counts, ns)])
    ax.set_ylim(0, 115)
    _style(ax, "Win Rate by Player Count", "Players", "Win Rate (%)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 2: evil win reason breakdown ────────────────────────────────────────

def plot_evil_breakdown(data: dict, out: str):
    counts = sorted(data)
    quest_f, reject_f, assn_f, other_f = [], [], [], []
    ns_evil = []
    for n in counts:
        evil = [e for e in data[n] if e.get("winner") == "evil"]
        total = len(evil) or 1
        ns_evil.append(len(evil))
        quest_f.append(sum(1 for e in evil if e.get("reason") == "3_quest_failures")   / total * 100)
        reject_f.append(sum(1 for e in evil if e.get("reason") == "5_consecutive_rejections") / total * 100)
        assn_f.append(sum(1 for e in evil if e.get("reason") == "merlin_assassinated") / total * 100)
        rest = 100 - quest_f[-1] - reject_f[-1] - assn_f[-1]
        other_f.append(max(rest, 0))

    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x, quest_f,  color=QUEST_COLOR, label="3 quest failures",        zorder=3)
    b2 = ax.bar(x, reject_f, color=REJCT_COLOR, label="5 consecutive rejections", zorder=3,
                bottom=quest_f)
    bot3 = [a + b for a, b in zip(quest_f, reject_f)]
    b3 = ax.bar(x, assn_f, color=ASSN_COLOR, label="Merlin assassinated", zorder=3, bottom=bot3)
    bot4 = [a + b for a, b in zip(bot3, assn_f)]
    if any(v > 0.5 for v in other_f):
        ax.bar(x, other_f, color="#BDC3C7", label="Other", zorder=3, bottom=bot4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}p\n(n={e})" for n, e in zip(counts, ns_evil)])
    ax.set_ylim(0, 110)
    _style(ax, "How Evil Wins — Breakdown by Player Count",
           "Players", "% of evil victories")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 3: merlin safety ─────────────────────────────────────────────────────

def plot_merlin_safety(data: dict, out: str):
    """Of games that reach the assassination phase, how often is Merlin caught?"""
    counts = sorted(data)
    caught_r, safe_r, ns_reach = [], [], []
    for n in counts:
        # games where assassination phase occurred = either merlin_assassinated or merlin_saved
        reach = [e for e in data[n]
                 if e.get("reason") in ("merlin_assassinated", "merlin_saved")]
        total = len(reach) or 1
        ns_reach.append(len(reach))
        caught = sum(1 for e in reach if e.get("reason") == "merlin_assassinated")
        caught_r.append(caught / total * 100)
        safe_r.append(100 - caught / total * 100)

    x = np.arange(len(counts))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, caught_r, w, color=EVIL_COLOR, label="Merlin caught",  zorder=3)
    ax.bar(x + w/2, safe_r,   w, color=GOOD_COLOR, label="Merlin escaped", zorder=3)
    for i, (c, s, n) in enumerate(zip(caught_r, safe_r, ns_reach)):
        ax.text(i - w/2, c + 2, f"{c:.0f}%", ha="center", fontsize=8)
        ax.text(i + w/2, s + 2, f"{s:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}p\n(reach={ep})" for n, ep in zip(counts, ns_reach)])
    ax.set_ylim(0, 115)
    ax.axhline(50, color="#AAAAAA", linewidth=1, linestyle="--", zorder=2)
    _style(ax, "Merlin Safety (games reaching assassination phase)",
           "Players", "% of assassination-phase games")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 4: overview (3-panel) ────────────────────────────────────────────────

def plot_overview(data: dict, out: str):
    counts = sorted(data)
    good_r, evil_r, ns = [], [], []
    quest_f, reject_f, assn_f = [], [], []
    caught_r, ns_reach = [], []

    for n in counts:
        eps   = data[n]; total = len(eps) or 1
        good  = sum(1 for e in eps if e.get("winner") == "good")
        good_r.append(good / total * 100); evil_r.append((total - good) / total * 100)
        ns.append(total)
        evil  = [e for e in eps if e.get("winner") == "evil"]; et = len(evil) or 1
        quest_f.append(sum(1 for e in evil if e.get("reason") == "3_quest_failures") / et * 100)
        reject_f.append(sum(1 for e in evil if e.get("reason") == "5_consecutive_rejections") / et * 100)
        assn_f.append(sum(1 for e in evil if e.get("reason") == "merlin_assassinated") / et * 100)
        reach  = [e for e in eps if e.get("reason") in ("merlin_assassinated", "merlin_saved")]
        rt = len(reach) or 1; ns_reach.append(len(reach))
        caught_r.append(sum(1 for e in reach if e.get("reason") == "merlin_assassinated") / rt * 100)

    x = np.arange(len(counts)); w = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Avalon Baseline Results", fontsize=14, fontweight="bold")

    # Panel 1: win rates
    ax = axes[0]
    ax.bar(x - w/2, good_r, w, color=GOOD_COLOR, label="Good", zorder=3)
    ax.bar(x + w/2, evil_r, w, color=EVIL_COLOR, label="Evil", zorder=3)
    ax.axhline(50, color="#AAAAAA", linewidth=1, linestyle="--", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}p" for n in counts])
    ax.set_ylim(0, 115); _style(ax, "Win Rates", "Players", "Win Rate (%)")
    ax.legend(frameon=False, fontsize=9)

    # Panel 2: evil breakdown
    ax = axes[1]
    ax.bar(x, quest_f,  color=QUEST_COLOR, label="Quest failures", zorder=3)
    ax.bar(x, reject_f, color=REJCT_COLOR, label="5 rejections",   zorder=3, bottom=quest_f)
    bot = [a + b for a, b in zip(quest_f, reject_f)]
    ax.bar(x, assn_f, color=ASSN_COLOR, label="Assassination", zorder=3, bottom=bot)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}p" for n in counts])
    ax.set_ylim(0, 110); _style(ax, "Evil Win Path", "Players", "% of evil victories")
    ax.legend(frameon=False, fontsize=9)

    # Panel 3: merlin safety
    ax = axes[2]
    ax.bar(x - w/2, caught_r, w, color=EVIL_COLOR, label="Caught",  zorder=3)
    ax.bar(x + w/2, [100 - c for c in caught_r], w, color=GOOD_COLOR, label="Escaped", zorder=3)
    ax.axhline(50, color="#AAAAAA", linewidth=1, linestyle="--", zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([f"{n}p" for n in counts])
    ax.set_ylim(0, 115); _style(ax, "Merlin Safety", "Players", "% of assassination games")
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/avalon")
    ap.add_argument("--out",     default="plots/avalon")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = load_summaries(args.log_dir)
    if not data:
        print("No summaries found. Run episodes first."); return

    for n, eps in sorted(data.items()):
        good = sum(1 for e in eps if e.get("winner") == "good")
        print(f"  {n}p: {len(eps)} eps — good {good}, evil {len(eps)-good}")

    plot_win_rates(data,     os.path.join(args.out, "win_rates.png"))
    plot_evil_breakdown(data, os.path.join(args.out, "evil_breakdown.png"))
    plot_merlin_safety(data,  os.path.join(args.out, "merlin_safety.png"))
    plot_overview(data,       os.path.join(args.out, "overview.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
