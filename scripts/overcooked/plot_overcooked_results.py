"""Plot Overcooked experiment results across layouts.

Usage:
    python scripts/overcooked/plot_overcooked_results.py [--log-dir logs/overcooked] [--out plots/overcooked]

Reads episode_*.summary.json (and episode_*.jsonl for scoring events)
from logs/overcooked/{run_id}/ and produces:
    score_distribution.png   — score boxplot by layout
    zero_score_rate.png      — fraction of episodes with zero deliveries by layout
    scoring_events.png       — when during the episode deliveries occur (JSONL)
    action_distribution.png  — action type breakdown by layout (JSONL)
    overview.png             — combined summary
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

LAYOUT_CONFIGS = {
    "cramped_room":       ("overcooked_noop_2p",                    "#4C9BE8"),
    "asymmetric_adv":     ("overcooked_noop_asymmetric_adv_2p",     "#E8604C"),
    "coordination_ring":  ("overcooked_noop_coordination_ring_2p",  "#4CAF7D"),
}
LAYOUT_LABELS = {
    "cramped_room":      "Cramped\nRoom",
    "asymmetric_adv":    "Asymmetric\nAdv",
    "coordination_ring": "Coordination\nRing",
}
GRID_COLOR = "#DDDDDD"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_summaries(log_dir: str) -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for layout, (run_id, _) in LAYOUT_CONFIGS.items():
        run_dir = os.path.join(log_dir, run_id)
        eps = []
        for path in glob.glob(os.path.join(run_dir, "episode_*.summary.json")):
            try:
                with open(path) as f:
                    s = json.load(f)
                info = s.get("final_info", {})
                if "score" in info:
                    eps.append(info)
            except (json.JSONDecodeError, OSError):
                continue
        if eps:
            data[layout] = eps
    return data


def load_step_events(log_dir: str) -> dict[str, list[dict]]:
    """Return {layout: [{step, score_delta, score}]} from JSONL files."""
    data: dict[str, list[dict]] = defaultdict(list)
    for layout, (run_id, _) in LAYOUT_CONFIGS.items():
        run_dir = os.path.join(log_dir, run_id)
        for jpath in sorted(glob.glob(os.path.join(run_dir, "episode_*.jsonl"))):
            ep_events = []
            seen_turns = set()
            try:
                with open(jpath) as f:
                    for line in f:
                        step = json.loads(line.strip())
                        turn = step.get("turn", 0)
                        if turn in seen_turns:
                            continue
                        seen_turns.add(turn)
                        info  = step.get("info", {})
                        score = info.get("score", 0)
                        delta = info.get("score_delta", step.get("reward", 0))
                        act   = str(step.get("parsed_action", "STAY")).upper()
                        ep_events.append({"step": turn, "score": score,
                                          "score_delta": delta, "action": act})
            except (json.JSONDecodeError, OSError):
                continue
            if ep_events:
                data[layout].append(ep_events)
    return dict(data)


# ── plot 1: score distribution ────────────────────────────────────────────────

def plot_score_distribution(data: dict, out: str):
    layouts = [l for l in LAYOUT_CONFIGS if l in data]
    scores_by_l = [[e["score"] for e in data[l]] for l in layouts]
    colors = [LAYOUT_CONFIGS[l][1] for l in layouts]

    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(42)
    bp = ax.boxplot(
        scores_by_l, positions=range(len(layouts)), widths=0.45,
        patch_artist=True, medianprops=dict(color="black", linewidth=2),
        zorder=3,
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col + "55")
        patch.set_edgecolor(col)
    for element in ("whiskers", "caps"):
        for item, col in zip(
            [bp[element][i*2:i*2+2] for i in range(len(layouts))], colors
        ):
            for line in item:
                line.set_color(col)
    for i, (scores, col) in enumerate(zip(scores_by_l, colors)):
        jit = rng.uniform(-0.15, 0.15, len(scores))
        ax.scatter(np.full(len(scores), i) + jit, scores,
                   color=col, alpha=0.5, s=18, zorder=4)
    ax.set_xticks(range(len(layouts)))
    ax.set_xticklabels([f"{LAYOUT_LABELS[l]}\n(n={len(data[l])})" for l in layouts])
    _style(ax, "Final Score Distribution by Layout", "Layout", "Dishes delivered")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 2: zero-score rate ───────────────────────────────────────────────────

def plot_zero_score_rate(data: dict, out: str):
    layouts = [l for l in LAYOUT_CONFIGS if l in data]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(layouts))
    zero_rates  = []
    any_rates   = []
    mean_scores = []
    ns = []
    for l in layouts:
        eps = data[l]; t = len(eps) or 1
        zeros   = sum(1 for e in eps if e["score"] == 0)
        zero_rates.append(zeros / t * 100)
        any_rates.append((t - zeros) / t * 100)
        mean_scores.append(np.mean([e["score"] for e in eps]))
        ns.append(t)

    colors = [LAYOUT_CONFIGS[l][1] for l in layouts]
    bars = ax.bar(x, zero_rates, color=["#E8604C"] * len(layouts), label="Zero deliveries", zorder=3)
    ax.bar(x, any_rates, bottom=zero_rates, color=colors, label="≥1 delivery", zorder=3)
    for i, (zr, ms) in enumerate(zip(zero_rates, mean_scores)):
        ax.text(i, zr / 2, f"{zr:.0f}%", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        ax.text(i, 103, f"μ={ms:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LAYOUT_LABELS[l]}\n(n={n})" for l, n in zip(layouts, ns)])
    ax.set_ylim(0, 112)
    _style(ax, "Coordination Success Rate by Layout",
           "Layout", "% of episodes")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 3: scoring event timing (from JSONL) ─────────────────────────────────

def plot_scoring_events(step_data: dict, out: str):
    """Histogram of step at which each delivery occurs."""
    layouts = [l for l in LAYOUT_CONFIGS if l in step_data]
    if not layouts:
        print("No JSONL data — skipping scoring_events.png"); return

    fig, axes = plt.subplots(1, len(layouts), figsize=(5 * len(layouts), 4), sharey=False)
    if len(layouts) == 1:
        axes = [axes]

    for ax, l in zip(axes, layouts):
        col = LAYOUT_CONFIGS[l][1]
        delivery_steps = []
        for ep in step_data[l]:
            for ev in ep:
                if ev.get("score_delta", 0) > 0:
                    delivery_steps.append(ev["step"])
        if delivery_steps:
            ax.hist(delivery_steps, bins=20, color=col, alpha=0.8, edgecolor="white")
            ax.axvline(np.median(delivery_steps), color="black", linewidth=1.5,
                       linestyle="--", label=f"median={np.median(delivery_steps):.0f}")
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(0.5, 0.5, "No deliveries", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")
        ax.set_title(LAYOUT_LABELS[l], fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Delivery count")
    fig.suptitle("When Deliveries Happen During Episodes", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


# ── plot 4: action distribution ───────────────────────────────────────────────

def plot_action_distribution(step_data: dict, out: str):
    layouts = [l for l in LAYOUT_CONFIGS if l in step_data]
    if not layouts:
        print("No JSONL data — skipping action_distribution.png"); return

    ACTION_GROUPS = {
        "MOVE":     ["MOVE N", "MOVE S", "MOVE E", "MOVE W"],
        "INTERACT": ["INTERACT"],
        "PICK_UP":  ["PICK_UP"],
        "DROP":     ["DROP"],
        "STAY":     ["STAY"],
    }
    GROUP_COLORS = {
        "MOVE":     "#4C9BE8",
        "INTERACT": "#4CAF7D",
        "PICK_UP":  "#F5A623",
        "DROP":     "#9B59B6",
        "STAY":     "#95A5A6",
    }
    groups = list(ACTION_GROUPS)
    x = np.arange(len(layouts))
    w = 0.7 / len(groups)

    fig, ax = plt.subplots(figsize=(9, 5))
    for gi, grp in enumerate(groups):
        fracs = []
        for l in layouts:
            all_acts = [ev["action"] for ep in step_data[l] for ev in ep]
            if not all_acts:
                fracs.append(0); continue
            grp_acts = ACTION_GROUPS[grp]
            count = sum(1 for a in all_acts
                        if any(a.startswith(ga) for ga in grp_acts))
            fracs.append(count / len(all_acts) * 100)
        offset = (gi - len(groups) / 2 + 0.5) * w
        ax.bar(x + offset, fracs, w, color=GROUP_COLORS[grp], label=grp, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([LAYOUT_LABELS[l] for l in layouts])
    _style(ax, "Action Distribution by Layout", "Layout", "% of all actions")
    ax.legend(frameon=False, fontsize=9, ncol=len(groups))
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/overcooked")
    ap.add_argument("--out",     default="plots/overcooked")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data      = load_summaries(args.log_dir)
    step_data = load_step_events(args.log_dir)

    if not data:
        print("No summaries found. Run episodes first."); return

    for l, eps in data.items():
        mean_s = np.mean([e["score"] for e in eps])
        zeros  = sum(1 for e in eps if e["score"] == 0)
        print(f"  {l}: {len(eps)} eps — mean score {mean_s:.2f}, zero-score {zeros}")

    plot_score_distribution(data,  os.path.join(args.out, "score_distribution.png"))
    plot_zero_score_rate(data,     os.path.join(args.out, "zero_score_rate.png"))
    plot_scoring_events(step_data, os.path.join(args.out, "scoring_events.png"))
    plot_action_distribution(step_data, os.path.join(args.out, "action_distribution.png"))
    print(f"\nAll plots → {args.out}/")


if __name__ == "__main__":
    main()
