#!/usr/bin/env python3
"""Game-outcome analysis across persona steering conditions.

For each steering condition, computes win/draw/loss rates (debate) or wolf
win rate (mafia) and tests significance vs the noop baseline.

Saves:
  reports/figures/debate/outcomes.png
  reports/figures/mafia/outcomes.png

Usage
-----
python scripts/textarena/outcome_analysis.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

# ── design tokens ─────────────────────────────────────────────────────────────

WIN_COLOR  = "#1baf7a"
DRAW_COLOR = "#eda100"
LOSS_COLOR = "#e34948"
NOOP_COLOR = "#9a9890"
SURFACE    = "#f5f4f0"
INK        = "#1a1917"
INK_MUTED  = "#6b6860"
GRID_C     = "#dddbd5"

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "semibold",
    "axes.labelsize":     9,
    "axes.labelcolor":    INK_MUTED,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "axes.grid":          True,
    "grid.color":         GRID_C,
    "grid.linewidth":     0.7,
    "xtick.color":        INK_MUTED,
    "ytick.color":        INK,
    "ytick.labelsize":    9,
    "xtick.labelsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  SURFACE,
    "figure.facecolor":   SURFACE,
    "axes.facecolor":     "#ffffff",
    "legend.fontsize":    8.5,
    "legend.frameon":     False,
})


# ── stats helpers ─────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def chi2_vs_noop(cond: Dict, noop: Dict, cats: List) -> Tuple[float, float]:
    obs  = [cond.get(c, 0) for c in cats]
    base = [noop.get(c, 0) for c in cats]
    if sum(obs) == 0 or sum(base) == 0:
        return float("nan"), float("nan")
    table = np.array([obs, base])
    mask  = table.sum(axis=0) > 0
    table = table[:, mask]
    if table.shape[1] < 2:
        return float("nan"), float("nan")
    chi2, p, *_ = chi2_contingency(table)
    return chi2, p


def sig_marker(p: float) -> str:
    if math.isnan(p): return ""
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return "n.s."


# ── data loading ──────────────────────────────────────────────────────────────

def load_debate_outcomes(run_dir: str) -> List[float]:
    rewards = []
    for p in sorted(glob.glob(os.path.join(run_dir, "episode_*.summary.json"))):
        try:
            s = json.load(open(p, encoding="utf-8"))
            r = s["final_rewards"].get("player_1")
            if r is not None:
                rewards.append(float(r))
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    return rewards


def load_mafia_wolf_outcomes(run_dir: str) -> List[float]:
    rewards = []
    for p in sorted(glob.glob(os.path.join(run_dir, "episode_*.summary.json"))):
        try:
            s = json.load(open(p, encoding="utf-8"))
            close = s["final_info"]["close_info"]
            for idx, info in close.items():
                if info.get("role") == "Mafia":
                    r = s["final_rewards"].get(f"player_{idx}")
                    if r is not None:
                        rewards.append(float(r))
                    break
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    return rewards


def counts(rewards: List[float], cats: List[float]) -> Dict[float, int]:
    c = Counter(rewards)
    return {cat: c.get(cat, 0) for cat in cats}


# ── debate figure ─────────────────────────────────────────────────────────────

def plot_debate_outcomes(conditions: Dict[str, List[float]],
                         noop_label: str, out_path: str) -> None:
    """Horizontal stacked-proportion bars, sorted by win rate, with CI + stars."""
    cats = [1.0, 0.0, -1.0]
    noop_rw  = conditions[noop_label]
    noop_cnt = counts(noop_rw, cats)
    noop_n   = len(noop_rw)

    # Sort steered conditions by win rate descending; noop always at top
    other = sorted(
        (k for k in conditions if k != noop_label),
        key=lambda k: counts(conditions[k], cats)[1.0] / len(conditions[k]),
        reverse=True,
    )
    labels = [noop_label] + other
    n_rows = len(labels)

    # Precompute per-row stats
    rows_data = []
    for label in labels:
        rw  = conditions[label]
        n   = len(rw)
        cnt = counts(rw, cats)
        wr  = cnt[1.0] / n if n else 0
        dr  = cnt[0.0] / n if n else 0
        lr  = cnt[-1.0] / n if n else 0
        lo, hi = wilson_ci(cnt[1.0], n)
        if label == noop_label:
            p = float("nan")
        else:
            _, p = chi2_vs_noop(cnt, noop_cnt, cats)
        rows_data.append(dict(label=label, n=n, wr=wr, dr=dr, lr=lr,
                              ci_lo=lo, ci_hi=hi, p=p))

    ys = np.arange(n_rows)
    h  = 0.58

    fig, ax = plt.subplots(figsize=(10, 0.65 * n_rows + 2.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor("#ffffff")
    ax.set_axisbelow(True)

    # Horizontal stacked bars
    for i, d in enumerate(rows_data):
        y = ys[i]
        ax.barh(y, d["wr"], h, color=WIN_COLOR,  zorder=2)
        ax.barh(y, d["dr"], h, left=d["wr"],            color=DRAW_COLOR, zorder=2)
        ax.barh(y, d["lr"], h, left=d["wr"] + d["dr"],  color=LOSS_COLOR, zorder=2)

        # 2 px white dividers between segments
        for edge in [d["wr"], d["wr"] + d["dr"]]:
            ax.barh(y, 0.004, h, left=edge - 0.002,
                    color="#ffffff", zorder=3, linewidth=0)

        # CI whiskers on win rate
        ax.errorbar(d["wr"], y,
                    xerr=[[d["wr"] - d["ci_lo"]], [d["ci_hi"] - d["wr"]]],
                    fmt="none", color=INK, capsize=3.5,
                    capthick=1.2, elinewidth=1.2, zorder=5)

        # Win % label
        if d["wr"] > 0.09:
            ax.text(d["wr"] / 2, y, f"{d['wr']:.0%}",
                    ha="center", va="center", fontsize=7.5,
                    color="#ffffff", fontweight="bold", zorder=6)
        elif d["wr"] > 0:
            ax.text(d["wr"] + 0.013, y, f"{d['wr']:.0%}",
                    ha="left", va="center", fontsize=7.5,
                    color=WIN_COLOR, fontweight="bold", zorder=6)

        # Significance star
        marker = sig_marker(d["p"])
        if marker:
            col = INK if marker != "n.s." else INK_MUTED
            ax.text(1.03, y, marker, ha="left", va="center",
                    fontsize=9 if marker != "n.s." else 7.5,
                    color=col, transform=ax.get_yaxis_transform(), zorder=6)

    # Y-axis labels — noop stands out in bold
    tick_labels = []
    for d in rows_data:
        if d["label"] == noop_label:
            tick_labels.append(f"← noop baseline  (n={d['n']})")
        else:
            tick_labels.append(f"{d['label']}  (n={d['n']})")

    ax.set_yticks(ys)
    ax.set_yticklabels(tick_labels, fontsize=9)

    # Make noop row label bold
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[0].set_color(NOOP_COLOR)

    # Noop row band
    ax.axhspan(-0.5, 0.5, color=NOOP_COLOR, alpha=0.10, zorder=0)
    ax.axhline(0.5, color=NOOP_COLOR, lw=1.0, ls="--", alpha=0.4, zorder=1)

    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Proportion of episodes", labelpad=7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Debate  —  player_1 (AGAINST) outcomes by steering trait",
                 pad=12, loc="left", color=INK)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color=GRID_C)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    # Legend row
    legend_patches = [
        mpatches.Patch(color=WIN_COLOR,  label="Win  (player_1)"),
        mpatches.Patch(color=DRAW_COLOR, label="Draw"),
        mpatches.Patch(color=LOSS_COLOR, label="Loss"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              ncol=3, frameon=True, framealpha=0.92,
              edgecolor=GRID_C, facecolor=SURFACE, fontsize=8.5)

    ax.text(0, -0.06, "Whiskers: 95% Wilson CI on win rate",
            transform=ax.transAxes, fontsize=7, color=INK_MUTED)
    ax.text(1.0, -0.06,
            "* p<.05  ** p<.01  *** p<.001   (χ² vs noop, all categories)",
            transform=ax.transAxes, fontsize=7, color=INK_MUTED, ha="right")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── mafia figure ──────────────────────────────────────────────────────────────

def plot_mafia_outcomes(conditions: Dict[str, List[float]],
                        noop_label: str, out_path: str) -> None:
    """Horizontal dot-plot of wolf win rate, CI bands, Fisher significance."""
    cats     = [1.0, -1.0]
    noop_rw  = conditions[noop_label]
    noop_cnt = counts(noop_rw, cats)
    noop_n   = len(noop_rw)
    noop_wr  = noop_cnt[1.0] / noop_n if noop_n else 0

    other = sorted(
        (k for k in conditions if k != noop_label),
        key=lambda k: counts(conditions[k], cats)[1.0] / len(conditions[k]),
        reverse=True,
    )
    labels = [noop_label] + other
    n_rows = len(labels)

    rows_data = []
    for label in labels:
        rw  = conditions[label]
        n   = len(rw)
        cnt = counts(rw, cats)
        wr  = cnt[1.0] / n if n else 0
        lo, hi = wilson_ci(cnt[1.0], n)
        if label == noop_label:
            p = float("nan")
        else:
            _, p = fisher_exact(
                [[cnt[1.0], cnt[-1.0]], [noop_cnt[1.0], noop_cnt[-1.0]]]
            )
        rows_data.append(dict(label=label, n=n, wr=wr, ci_lo=lo, ci_hi=hi, p=p))

    ys = np.arange(n_rows)

    fig, ax = plt.subplots(figsize=(9, 0.65 * n_rows + 2.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor("#ffffff")
    ax.set_axisbelow(True)

    for i, d in enumerate(rows_data):
        y     = ys[i]
        is_np = d["label"] == noop_label
        color = NOOP_COLOR if is_np else WIN_COLOR

        # CI band
        ax.barh(y, d["ci_hi"] - d["ci_lo"], 0.15,
                left=d["ci_lo"], color=color, alpha=0.22, zorder=2)
        ax.plot([d["ci_lo"], d["ci_hi"]], [y, y],
                color=color, lw=1.8, alpha=0.55, zorder=3)

        # Dot
        ax.scatter(d["wr"], y, s=70, color=color, zorder=4, linewidths=0)

        # Value label
        ax.text(d["ci_hi"] + 0.012, y, f"{d['wr']:.0%}",
                va="center", ha="left", fontsize=9,
                color=color, fontweight="semibold")

        # Significance star
        marker = sig_marker(d["p"])
        if marker:
            col = INK if marker != "n.s." else INK_MUTED
            ax.text(1.03, y, marker, ha="left", va="center",
                    fontsize=9 if marker != "n.s." else 7.5,
                    color=col, transform=ax.get_yaxis_transform())

    tick_labels = []
    for d in rows_data:
        if d["label"] == noop_label:
            tick_labels.append(f"← noop baseline  (n={d['n']})")
        else:
            tick_labels.append(f"{d['label']}  (n={d['n']})")

    ax.set_yticks(ys)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[0].set_color(NOOP_COLOR)

    # Noop reference line + band
    ax.axvline(noop_wr, color=NOOP_COLOR, lw=1.4, ls="--", zorder=1,
               label=f"noop baseline  ({noop_wr:.0%})")
    ax.axhspan(-0.5, 0.5, color=NOOP_COLOR, alpha=0.10, zorder=0)
    ax.axhline(0.5, color=NOOP_COLOR, lw=1.0, ls="--", alpha=0.4, zorder=1)

    ax.invert_yaxis()
    x_max = max(d["ci_hi"] for d in rows_data) + 0.14
    ax.set_xlim(0, min(1.0, x_max))
    ax.set_xlabel("Wolf win rate", labelpad=7)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Mafia  —  wolf win rate by steering trait",
                 pad=12, loc="left", color=INK)

    ax.legend(loc="lower right", frameon=True, framealpha=0.92,
              edgecolor=GRID_C, facecolor=SURFACE)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color=GRID_C)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    ax.text(0, -0.06, "Dot: win rate  |  Line + band: 95% Wilson CI",
            transform=ax.transAxes, fontsize=7, color=INK_MUTED)
    ax.text(1.0, -0.06, "* p<.05   (Fisher's exact vs noop)",
            transform=ax.transAxes, fontsize=7, color=INK_MUTED, ha="right")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--out-dir",  default="reports/figures")
    args = ap.parse_args()

    # ── Debate ────────────────────────────────────────────────────────────────
    print("=== Debate ===")
    debate_base = os.path.join(args.logs_dir, "debate")
    debate_cond: Dict[str, List[float]] = {}

    for noop_id in ["debate_noop_probe_2p", "debate_noop_2p"]:
        rw = load_debate_outcomes(os.path.join(debate_base, noop_id))
        if rw:
            debate_cond["noop"] = rw
            print(f"  noop: {noop_id}  n={len(rw)}")
            break

    for run_dir in sorted(glob.glob(
            os.path.join(debate_base, "debate_activation_p1_*"))):
        m = re.search(r"debate_activation_p1_(.+)_2p$", os.path.basename(run_dir))
        if not m:
            continue
        trait = m.group(1)
        rw = load_debate_outcomes(run_dir)
        if rw:
            debate_cond[trait] = rw
            cnt = counts(rw, [1.0, 0.0, -1.0])
            n   = len(rw)
            print(f"  {trait:25s}  n={n}  "
                  f"win={cnt[1.0]/n:.0%} draw={cnt[0.0]/n:.0%} "
                  f"loss={cnt[-1.0]/n:.0%}")

    if "noop" in debate_cond and len(debate_cond) > 1:
        plot_debate_outcomes(debate_cond, "noop",
                             os.path.join(args.out_dir, "debate", "outcomes.png"))

    # ── Mafia ─────────────────────────────────────────────────────────────────
    print("=== Mafia ===")
    mafia_base = os.path.join(args.logs_dir, "mafia")
    mafia_cond: Dict[str, List[float]] = {}

    for noop_id in ["mafia_noop_probe_8p", "mafia_noop_8p"]:
        rw = load_mafia_wolf_outcomes(os.path.join(mafia_base, noop_id))
        if rw:
            mafia_cond["noop"] = rw
            cnt = counts(rw, [1.0, -1.0])
            n   = len(rw)
            print(f"  noop: {noop_id}  n={n}  wolf_win={cnt[1.0]/n:.0%}")
            break

    for run_dir in sorted(glob.glob(
            os.path.join(mafia_base, "mafia_activation_wolf_*"))):
        m = re.search(r"mafia_activation_wolf_(.+)_8p$", os.path.basename(run_dir))
        if not m:
            continue
        trait = m.group(1)
        rw = load_mafia_wolf_outcomes(run_dir)
        if rw:
            mafia_cond[trait] = rw
            cnt = counts(rw, [1.0, -1.0])
            n   = len(rw)
            print(f"  {trait:25s}  n={n}  wolf_win={cnt[1.0]/n:.0%}")

    if "noop" in mafia_cond and len(mafia_cond) > 1:
        plot_mafia_outcomes(mafia_cond, "noop",
                            os.path.join(args.out_dir, "mafia", "outcomes.png"))

    print("Done.")


if __name__ == "__main__":
    main()
