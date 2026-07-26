"""Persona-impact analysis and plots.

Usage (after merge_persona_cases.py):
    python scripts/plot_persona_impact.py
    python scripts/plot_persona_impact.py --results logs/persona_impact/case_results.jsonl

Outputs: plots/persona_impact/*.png  (6 figures)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── palette (from validated reference) ────────────────────────────────────────
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK2      = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

DIV_POS   = "#e34948"   # red pole  (guess higher than baseline)
DIV_NEG   = "#2a78d6"   # blue pole (guess lower)
DIV_MID   = "#f0efec"   # neutral gray midpoint

SEQ_LO    = "#cde2fb"
SEQ_HI    = "#0d366b"

# 5 categorical slots (adjacent-safe order from reference palette)
CAT_COLORS = {
    "oscillating":      "#2a78d6",   # slot 1 blue
    "stuck":            "#eb6834",   # slot 2 orange
    "near_convergence": "#1baf7a",   # slot 3 aqua
    "late_game":        "#eda100",   # slot 4 yellow
    "misc":             "#e87ba4",   # slot 5 magenta
    "first_round":      "#e87ba4",   # same as misc (only 1 case)
}
CAT_ORDER = ["oscillating", "stuck", "near_convergence", "late_game", "misc", "first_round"]

mpl.rcParams.update({
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BASELINE,
    "axes.labelcolor":   INK2,
    "axes.titlecolor":   INK,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "text.color":        INK,
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": SURFACE,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def load(results_path: Path) -> pd.DataFrame:
    records = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(records)
    df = df[df["action"].notna()].copy()
    df["action"] = df["action"].astype(float)
    df["delta"]  = df["action"] - df["original_action"].astype(float)
    return df


def cond_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition: mean_delta, std_delta, std_action, n, is_tom, base_name."""
    stats = (df.groupby(["condition", "condition_idx"])
               .agg(mean_delta=("delta",  "mean"),
                    std_delta =("delta",  "std"),
                    std_action=("action", "std"),
                    mean_action=("action","mean"),
                    n         =("action", "count"))
               .reset_index()
               .sort_values("condition_idx"))
    stats["is_tom"]   = stats["condition"].str.endswith("_tom")
    stats["base_name"]= stats["condition"].str.removesuffix("_tom")
    return stats


def diverging_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "div", [DIV_NEG, DIV_MID, DIV_POS])


# ── plot 1: mean delta per condition (ranked diverging bar) ───────────────────

def plot_mean_delta(df: pd.DataFrame, out: Path) -> None:
    stats = cond_stats(df).sort_values("mean_delta")
    n = len(stats)
    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.28)))

    vmax = stats["mean_delta"].abs().max() * 1.1 or 1
    cmap = diverging_cmap()
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    bars = ax.barh(range(n), stats["mean_delta"].values,
                   color=[cmap(norm(v)) for v in stats["mean_delta"].values],
                   height=0.7, linewidth=0)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(stats["condition"].values, fontsize=8)
    ax.set_xlabel("Mean Δ action  (persona − baseline)", color=INK2)
    ax.set_title("How much each persona shifts the agent's guess\n(negative = guesses lower, positive = higher)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # annotate ±2 biggest movers
    for i, (_, row) in enumerate(stats.iterrows()):
        if abs(row["mean_delta"]) >= stats["mean_delta"].abs().nlargest(4).min():
            ax.text(row["mean_delta"] + (0.2 if row["mean_delta"] >= 0 else -0.2),
                    i, f"{row['mean_delta']:.1f}", va="center",
                    ha="left" if row["mean_delta"] >= 0 else "right",
                    fontsize=7.5, color=INK2)

    fig.tight_layout()
    fig.savefig(out / "1_mean_delta_ranked.png")
    plt.close(fig)
    print(f"  saved 1_mean_delta_ranked.png  ({n} conditions)")


# ── plot 2: action variance per condition (erraticity) ───────────────────────

def plot_variance(df: pd.DataFrame, out: Path) -> None:
    stats = cond_stats(df).sort_values("std_action", ascending=False)
    n = len(stats)
    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.28)))

    # sequential blue: map std to [SEQ_LO, SEQ_HI]
    cmap = mcolors.LinearSegmentedColormap.from_list("seq", [SEQ_LO, SEQ_HI])
    vmin, vmax = stats["std_action"].min(), stats["std_action"].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    ax.barh(range(n), stats["std_action"].values,
            color=[cmap(norm(v)) for v in stats["std_action"].values],
            height=0.7, linewidth=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(stats["condition"].values, fontsize=8)
    ax.set_xlabel("Std of action across reps  (higher = more erratic)", color=INK2)
    ax.set_title("Action variance per condition\n(erraticity: how consistently does the persona guess?)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # mark plain baseline
    plain_std = stats.loc[stats["condition"] == "plain", "std_action"]
    if not plain_std.empty:
        ax.axvline(plain_std.values[0], color=DIV_NEG, linewidth=1.2,
                   linestyle=":", label="plain baseline")
        ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out / "2_action_variance.png")
    plt.close(fig)
    print(f"  saved 2_action_variance.png")


# ── plot 3: ToM effect — dumbbell ─────────────────────────────────────────────

def plot_tom_dumbbell(df: pd.DataFrame, out: Path) -> None:
    stats = cond_stats(df)
    base  = stats[~stats["is_tom"]].set_index("base_name")["mean_delta"]
    tom   = stats[ stats["is_tom"]].set_index("base_name")["mean_delta"]
    personas = sorted(set(base.index) & set(tom.index) - {"plain"})
    if not personas:
        print("  skipped 3_tom_dumbbell.png (no ToM pairs yet)")
        return

    personas_sorted = sorted(personas, key=lambda p: base.get(p, 0))
    n = len(personas_sorted)
    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.38)))

    for i, p in enumerate(personas_sorted):
        b, t = base.get(p, np.nan), tom.get(p, np.nan)
        if np.isnan(b) or np.isnan(t):
            continue
        ax.plot([b, t], [i, i], color=BASELINE, linewidth=1.5, zorder=1)
        ax.scatter([b], [i], color=DIV_NEG, s=55, zorder=2, label="base" if i == 0 else "")
        ax.scatter([t], [i], color=DIV_POS, s=55, zorder=2, marker="D",
                   label="+ToM" if i == 0 else "")

    ax.axvline(0, color=BASELINE, linewidth=1, linestyle="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels(personas_sorted, fontsize=8)
    ax.set_xlabel("Mean Δ action", color=INK2)
    ax.set_title("Theory of Mind effect per persona\n(circle = base, diamond = +ToM)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out / "3_tom_dumbbell.png")
    plt.close(fig)
    print(f"  saved 3_tom_dumbbell.png  ({n} persona pairs)")


# ── plot 4: category sensitivity — small multiples ────────────────────────────

def plot_category_breakdown(df: pd.DataFrame, out: Path) -> None:
    cats = [c for c in CAT_ORDER if c in df["category"].unique()]
    if not cats:
        return
    stats = cond_stats(df)
    cond_order = stats.sort_values("condition_idx")["condition"].tolist()

    n_cols = min(3, len(cats))
    n_rows = int(np.ceil(len(cats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             sharey=False, sharex=False)
    axes = np.array(axes).flatten()

    for ax, cat in zip(axes, cats):
        sub = df[df["category"] == cat]
        per_cond = sub.groupby("condition")["delta"].mean().reindex(cond_order)
        color = CAT_COLORS.get(cat, MUTED)
        cmap  = mcolors.LinearSegmentedColormap.from_list("cat", [DIV_MID, color])

        vmax = per_cond.abs().max() or 1
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        bar_colors = [cmap(norm(v)) if not np.isnan(v) else MUTED for v in per_cond.values]

        ax.bar(range(len(per_cond)), per_cond.fillna(0).values,
               color=bar_colors, width=0.8, linewidth=0)
        ax.axhline(0, color=BASELINE, linewidth=0.8)
        ax.set_xticks([])
        ax.set_title(cat.replace("_", " "), fontsize=10, fontweight="bold",
                     color=color)
        ax.set_ylabel("mean Δ action", fontsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        n_cases = sub["case_id"].nunique()
        ax.text(0.98, 0.02, f"n={n_cases} cases", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=MUTED)

    for ax in axes[len(cats):]:
        ax.set_visible(False)

    fig.suptitle("Persona impact by scenario category\n(each bar = one condition, sorted by condition index)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "4_category_breakdown.png")
    plt.close(fig)
    print(f"  saved 4_category_breakdown.png  ({len(cats)} categories)")


# ── plot 5: case × condition heatmap ─────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, out: Path) -> None:
    stats = cond_stats(df)
    cond_order = stats.sort_values("condition_idx")["condition"].tolist()

    # sort cases by category then case_id
    case_order = (df[["case_id", "category"]]
                  .drop_duplicates()
                  .sort_values(["category", "case_id"])["case_id"].tolist())

    pivot = (df.groupby(["case_id", "condition"])["delta"]
               .mean()
               .unstack(fill_value=np.nan)
               .reindex(index=case_order, columns=cond_order))

    vmax = np.nanpercentile(np.abs(pivot.values), 95) or 5
    cmap = diverging_cmap()

    fig, ax = plt.subplots(figsize=(max(10, len(cond_order) * 0.28),
                                    max(8, len(case_order) * 0.22)))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.6, label="mean Δ action")

    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_order, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(case_order)))
    ax.set_yticklabels(case_order, fontsize=7)
    ax.set_title("Mean Δ action per case × condition\n(blue = guesses lower, red = higher than baseline)",
                 fontsize=11, fontweight="bold", pad=10)

    # draw category dividers on y-axis
    cat_seq = df.set_index("case_id")["category"].to_dict()
    prev_cat = None
    for i, cid in enumerate(case_order):
        cat = cat_seq.get(cid, "")
        if cat != prev_cat and i > 0:
            ax.axhline(i - 0.5, color=INK, linewidth=0.6, alpha=0.4)
        prev_cat = cat

    fig.tight_layout()
    fig.savefig(out / "5_case_condition_heatmap.png")
    plt.close(fig)
    print(f"  saved 5_case_condition_heatmap.png  ({len(case_order)}×{len(cond_order)})")


# ── plot 6: oscillation focus — variance by category × plain vs persona ───────

def plot_oscillation_focus(df: pd.DataFrame, out: Path) -> None:
    """For oscillating cases: compare std(action) under plain vs all conditions."""
    osc = df[df["category"] == "oscillating"]
    if osc.empty:
        print("  skipped 6_oscillation_focus.png (no oscillating cases yet)")
        return

    stats = cond_stats(df)
    cond_order = stats.sort_values("condition_idx")["condition"].tolist()

    # per-condition std(action) restricted to oscillating cases
    per_cond = (osc.groupby("condition")["action"]
                   .std()
                   .reindex(cond_order)
                   .sort_values(ascending=False))

    plain_std = per_cond.get("plain", np.nan)

    n = len(per_cond.dropna())
    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.28)))

    cmap = mcolors.LinearSegmentedColormap.from_list("seq", [SEQ_LO, SEQ_HI])
    vmin, vmax = per_cond.min(), per_cond.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    vals   = per_cond.dropna()
    colors = [cmap(norm(v)) for v in vals.values]
    ax.barh(range(len(vals)), vals.values, color=colors, height=0.7, linewidth=0)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(vals.index, fontsize=8)
    ax.set_xlabel("Std of action  (oscillating cases only)", color=INK2)
    ax.set_title("Which personas reduce erratic behavior?\n(oscillating cases: lower std = more stable guessing)",
                 fontsize=11, fontweight="bold", pad=10)

    if not np.isnan(plain_std):
        ax.axvline(plain_std, color=DIV_POS, linewidth=1.4,
                   linestyle="--", label=f"plain baseline ({plain_std:.1f})")
        ax.legend(frameon=False, fontsize=8)

    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out / "6_oscillation_variance.png")
    plt.close(fig)
    print(f"  saved 6_oscillation_variance.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="logs/persona_impact/case_results.jsonl",
                        type=Path)
    parser.add_argument("--out", default="plots/persona_impact", type=Path)
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(f"Results not found: {args.results}\n"
                         "Run merge_persona_cases.py first.")

    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.results)

    n_conds  = df["condition"].nunique()
    n_cases  = df["case_id"].nunique()
    n_reps   = df.groupby(["case_id", "condition"])["rep"].count().max()
    parse_ok = df["action"].notna().mean() * 100

    print(f"Loaded {len(df)} records  |  {n_conds} conditions  |  {n_cases} cases  "
          f"|  max {n_reps} reps/cell  |  parse rate {parse_ok:.1f}%")
    print(f"Saving plots to {args.out}/\n")

    plot_mean_delta(df, args.out)
    plot_variance(df, args.out)
    plot_tom_dumbbell(df, args.out)
    plot_category_breakdown(df, args.out)
    plot_heatmap(df, args.out)
    plot_oscillation_focus(df, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
