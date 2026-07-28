"""Persona-impact analysis and plots.

Usage (after merge_persona_cases.py):
    python scripts/plot_persona_impact.py
    python scripts/plot_persona_impact.py --results logs/persona_impact/case_results.jsonl

Outputs: plots/persona_impact/*.png  (6 figures)
"""
from __future__ import annotations

import argparse
import json
import textwrap
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

    # Use per-case mean of the plain condition (20 reps) as baseline instead of
    # original_action (single game observation — too noisy).
    plain_baseline = (df[df["condition"] == "plain"]
                        .groupby("case_id")["action"]
                        .mean()
                        .rename("plain_baseline"))
    df = df.join(plain_baseline, on="case_id")

    # Fall back to original_action for any case where plain is missing
    df["plain_baseline"] = df["plain_baseline"].fillna(df["original_action"].astype(float))
    df["delta"] = df["action"] - df["plain_baseline"]
    return df


def cond_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition summary stats including decomposed variance."""
    stats = (df.groupby(["condition", "condition_idx"])
               .agg(mean_delta=("delta",  "mean"),
                    std_delta =("delta",  "std"),
                    std_action=("action", "std"),
                    mean_action=("action","mean"),
                    n         =("action", "count"))
               .reset_index()
               .sort_values("condition_idx"))

    # within-case std: average of per-case std across 20 reps.
    # This is pure model stochasticity (same prompt → different outputs).
    # std_action above also includes between-case variance, which is large
    # because 50 cases span very different game states and target ranges.
    within = (df.groupby(["condition", "case_id"])["action"]
                .std(ddof=1)
                .groupby("condition")
                .mean()
                .rename("within_case_std"))
    stats = stats.join(within, on="condition")

    stats["is_tom"]   = stats["condition"].str.endswith("_tom")
    stats["base_name"]= stats["condition"].str.removesuffix("_tom")
    return stats


BASELINES = {"plain", "plain_tom"}


def _bold_baseline_ticks(ax: "plt.Axes", labels: list[str], axis: str = "y") -> None:
    """Bold the tick labels for plain / plain_tom entries."""
    ticks = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for tick, lbl in zip(ticks, labels):
        if lbl in BASELINES:
            tick.set_fontweight("bold")
            tick.set_color(INK)


def _shade_baseline_bars(ax: "plt.Axes", labels: list[str]) -> None:
    """Add a faint horizontal band behind each baseline bar."""
    for i, lbl in enumerate(labels):
        if lbl in BASELINES:
            ax.axhspan(i - 0.45, i + 0.45, color=GRID, alpha=0.55, zorder=0)


def _shade_baseline_cols(ax: "plt.Axes", labels: list[str]) -> None:
    """Add a faint vertical band behind each baseline column in a heatmap."""
    for j, lbl in enumerate(labels):
        if lbl in BASELINES:
            ax.axvspan(j - 0.5, j + 0.5, color=GRID, alpha=0.55, zorder=0)


def diverging_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "div", [DIV_NEG, DIV_MID, DIV_POS])


# ── plot 00: case taxonomy infographic ───────────────────────────────────────

CASES_JSON = Path("cases/gbs/persona_impact_cases.json")

CAT_META = {
    "oscillating":      {"color": "#2a78d6", "label": "Oscillating",      "desc": "Agent repeatedly swings between high and low guesses, failing to settle"},
    "stuck":            {"color": "#eb6834", "label": "Stuck",             "desc": "Agent has converged to a wrong value and does not adjust despite feedback"},
    "near_convergence": {"color": "#1baf7a", "label": "Near Convergence",  "desc": "Agent is close to the target but making small residual errors"},
    "late_game":        {"color": "#eda100", "label": "Late Game",         "desc": "Deep into the episode (round ≥ 18); coordination under pressure"},
    "misc":             {"color": "#9b6bbf", "label": "Misc / Mid-Game",   "desc": "Random mid-game states for variety; no single failure mode"},
    "first_round":      {"color": "#7a6952", "label": "First Round",       "desc": "Round 1 baseline: agent has no prior game context"},
}
CAT_ORDER_TAX = ["oscillating", "stuck", "near_convergence", "late_game", "misc", "first_round"]


def plot_case_taxonomy(out: Path) -> None:
    """Infographic: 6 category cards, 1 example each, total count prominent."""
    if not CASES_JSON.exists():
        print("  skipped 00_case_taxonomy.png (cases JSON not found)")
        return

    cases = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    by_cat: dict[str, list] = {c: [] for c in CAT_ORDER_TAX}
    for case in cases:
        by_cat.setdefault(case["category"], []).append(case)

    total_cases = sum(len(v) for v in by_cat.values())

    # 3-col × 2-row card grid
    n_cols, n_rows = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 10))
    fig.patch.set_facecolor(SURFACE)

    fig.suptitle(
        f"Case Taxonomy — {total_cases} Game States across 6 Behavioural Categories",
        fontsize=15, fontweight="bold", color=INK, y=1.01)
    fig.text(0.5, 0.995,
             "Each case is a mid-episode snapshot; the agent sees the full history "
             "and must output FINAL GUESS: [0–50].",
             ha="center", va="top", fontsize=9, color=INK2)

    for idx, cat in enumerate(CAT_ORDER_TAX):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(SURFACE)

        meta  = CAT_META.get(cat, {"color": MUTED, "label": cat, "desc": ""})
        color = meta["color"]
        label = meta["label"]
        desc  = meta["desc"]
        items = by_cat.get(cat, [])
        n     = len(items)

        # ── coloured header band ──────────────────────────────────────────────
        ax.add_patch(plt.Rectangle((0, 0.72), 1, 0.28,
                                   facecolor=color, alpha=0.18,
                                   transform=ax.transAxes, clip_on=True))
        # thick left accent bar
        ax.add_patch(plt.Rectangle((0, 0), 0.025, 1,
                                   facecolor=color, alpha=0.9,
                                   transform=ax.transAxes, clip_on=True))

        # category name + count
        ax.text(0.055, 0.885, label,
                transform=ax.transAxes, fontsize=13, fontweight="bold",
                color=color, va="center")
        ax.text(0.055, 0.775, f"{n} {'case' if n == 1 else 'cases'}",
                transform=ax.transAxes, fontsize=10, color=color,
                va="center", alpha=0.85)

        # description
        wrapped_desc = textwrap.fill(desc, width=52)
        ax.text(0.055, 0.635, wrapped_desc,
                transform=ax.transAxes, fontsize=8.5, color=INK2,
                va="top", style="italic", linespacing=1.5)

        # ── example case box ─────────────────────────────────────────────────
        if items:
            ex   = items[0]
            cid  = ex["case_id"]
            rnd  = ex["round"]
            tgt  = ex["target"]
            orig = ex["original_action"]

            # light box background
            ax.add_patch(plt.Rectangle((0.055, 0.04), 0.93, 0.36,
                                       facecolor=color, alpha=0.07,
                                       transform=ax.transAxes,
                                       linewidth=0.8,
                                       edgecolor=color))

            ax.text(0.075, 0.365, "Example case",
                    transform=ax.transAxes, fontsize=7,
                    color=color, fontweight="bold",
                    fontfamily="sans-serif", alpha=0.9)

            ax.text(0.075, 0.295, cid,
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                    color=INK, va="center", fontfamily="monospace")

            ax.text(0.075, 0.215,
                    f"Round {rnd}   ·   Target {tgt}   ·   Original guess: {orig}",
                    transform=ax.transAxes, fontsize=9, color=INK2, va="center")

            # action bar (0–50)
            bar_left, bar_right = 0.075, 0.97
            bar_y = 0.11
            bar_w = bar_right - bar_left
            filled = bar_left + (orig / 50) * bar_w

            ax.plot([bar_left, bar_right], [bar_y, bar_y],
                    color=GRID, linewidth=4, solid_capstyle="round",
                    transform=ax.transAxes)
            ax.plot([bar_left, filled], [bar_y, bar_y],
                    color=color, linewidth=4, alpha=0.75,
                    solid_capstyle="round", transform=ax.transAxes)
            ax.scatter([filled], [bar_y], s=55, color=color, zorder=3,
                       linewidth=0, transform=ax.transAxes)
            ax.text(bar_left - 0.01, bar_y, "0",
                    transform=ax.transAxes, fontsize=7, color=MUTED,
                    ha="right", va="center")
            ax.text(bar_right + 0.01, bar_y, "50",
                    transform=ax.transAxes, fontsize=7, color=MUTED,
                    ha="left", va="center")
            ax.text(filled, bar_y + 0.07, str(orig),
                    transform=ax.transAxes, fontsize=8, color=color,
                    ha="center", va="bottom", fontweight="bold")

        # outer card border
        for spine_pos in ["top", "right", "bottom", "left"]:
            ax.spines[spine_pos].set_visible(False)
        rect = plt.Rectangle((0, 0), 1, 1, fill=False,
                              edgecolor=GRID, linewidth=1.2,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

    fig.tight_layout(pad=1.4)
    fig.savefig(out / "00_case_taxonomy.png", dpi=130, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print("  saved 00_case_taxonomy.png")


# ── plot 1: mean delta per condition (ranked diverging bar) ───────────────────

def plot_mean_delta(df: pd.DataFrame, out: Path) -> None:
    stats = cond_stats(df).sort_values("condition")
    n = len(stats)
    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.28)))

    vmax = stats["mean_delta"].abs().max() * 1.1 or 1
    cmap = diverging_cmap()
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    labels = stats["condition"].tolist()
    _shade_baseline_bars(ax, labels)
    bars = ax.barh(range(n), stats["mean_delta"].values,
                   color=[cmap(norm(v)) for v in stats["mean_delta"].values],
                   height=0.7, linewidth=0)
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    _bold_baseline_ticks(ax, labels)
    ax.set_xlabel("Mean Δ action  (persona − baseline)", color=INK2)
    ax.set_title("How much each persona shifts the agent's guess\n(negative = guesses lower, positive = higher)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

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
    # Use within_case_std: std of 20 reps on the *same* case, averaged over cases.
    # This isolates model stochasticity from between-case variance.
    # (std_action is large for ALL conditions because 50 cases span very different
    # game states — a case with target 40 and one with target 8 will produce
    # vastly different actions even under the same persona. That's not erraticity,
    # that's just case diversity. within_case_std strips that out.)
    stats = cond_stats(df).sort_values("condition")
    n = len(stats)
    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.28)))

    cmap = mcolors.LinearSegmentedColormap.from_list("seq", [SEQ_LO, SEQ_HI])
    vmin = stats["within_case_std"].min()
    vmax = stats["within_case_std"].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    labels = stats["condition"].tolist()
    _shade_baseline_bars(ax, labels)
    ax.barh(range(n), stats["within_case_std"].values,
            color=[cmap(norm(v)) for v in stats["within_case_std"].values],
            height=0.7, linewidth=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    _bold_baseline_ticks(ax, labels)
    ax.set_xlabel("Mean within-case std  (avg of per-case std across 20 reps)", color=INK2)
    ax.set_title("Model stochasticity per condition\n"
                 "(within-case std: same prompt → how much does the action vary?)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plain_std = stats.loc[stats["condition"] == "plain", "within_case_std"]
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

    personas_sorted = sorted(personas)
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

    _shade_baseline_cols(ax, cond_order)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_order, rotation=60, ha="right", fontsize=7)
    _bold_baseline_ticks(ax, cond_order, axis="x")
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
    """For oscillating cases: within-case std per condition (pure model stochasticity)."""
    osc = df[df["category"] == "oscillating"]
    if osc.empty:
        print("  skipped 6_oscillation_variance.png (no oscillating cases yet)")
        return

    stats = cond_stats(df)
    cond_order = stats.sort_values("condition_idx")["condition"].tolist()

    # within-case std restricted to oscillating cases
    within = (osc.groupby(["condition", "case_id"])["action"]
                 .std(ddof=1)
                 .groupby("condition")
                 .mean()
                 .reindex(sorted(cond_order)))

    plain_std = within.get("plain", np.nan)

    vals = within.dropna()
    n    = len(vals)
    fig, ax = plt.subplots(figsize=(8, max(5, n * 0.28)))

    cmap   = mcolors.LinearSegmentedColormap.from_list("seq", [SEQ_LO, SEQ_HI])
    norm   = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
    labels = vals.index.tolist()
    _shade_baseline_bars(ax, labels)
    ax.barh(range(n), vals.values,
            color=[cmap(norm(v)) for v in vals.values], height=0.7, linewidth=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    _bold_baseline_ticks(ax, labels)
    ax.set_xlabel("Mean within-case std  (oscillating cases; avg of per-case std over 20 reps)",
                  color=INK2)
    ax.set_title("Which personas stabilise oscillating agents?\n"
                 "(within-case std on oscillating cases only — lower = more consistent per state)",
                 fontsize=11, fontweight="bold", pad=10)

    if not np.isnan(plain_std):
        ax.axvline(plain_std, color=DIV_POS, linewidth=1.4,
                   linestyle="--", label=f"plain ({plain_std:.1f})")
        ax.legend(frameon=False, fontsize=8)

    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out / "6_oscillation_variance.png")
    plt.close(fig)
    print(f"  saved 6_oscillation_variance.png")


# ── plot 7: per-case violins (20 raw action samples each) ─────────────────────

def plot_violin_scatter(df: pd.DataFrame, out: Path) -> None:
    """For plain + top-5 personas: one mini-violin per case (20 raw action samples).

    Layout: columns = conditions, rows = cases (sorted by original_action).
    Each violin shows the true action distribution for that game state under
    that persona, so you can see shape, spread, and multimodality directly.
    """
    stats   = cond_stats(df)
    non_tom = stats[~stats["is_tom"] & ~stats["condition"].isin(("plain", "plain_tom"))]
    top5    = non_tom.nlargest(3, "mean_delta")["condition"].tolist() + \
              non_tom.nsmallest(2, "mean_delta")["condition"].tolist()
    baselines   = [c for c in ("plain", "plain_tom") if c in df["condition"].unique()]
    show_conds  = baselines + top5

    case_order = df["case_id"].drop_duplicates().tolist()
    n_cases    = len(case_order)
    n_conds    = len(show_conds)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, n_conds,
                             figsize=(3.8 * n_conds, max(10, n_cases * 0.32)),
                             sharey=True)
    if n_conds == 1:
        axes = [axes]

    for col, (ax, cond) in enumerate(zip(axes, show_conds)):
        sub      = df[df["condition"] == cond]
        mean_d   = stats.loc[stats["condition"] == cond, "mean_delta"]
        fill_col = DIV_POS if (not mean_d.empty and mean_d.values[0] >= 0) else DIV_NEG
        if cond in ("plain", "plain_tom"):
            fill_col = MUTED

        for i, case_id in enumerate(case_order):
            actions = sub[sub["case_id"] == case_id]["action"].dropna().values
            if len(actions) < 2:
                continue

            if actions.std() > 1e-9:
                try:
                    parts = ax.violinplot([actions], positions=[i], vert=False,
                                         showmedians=True, showextrema=False,
                                         widths=0.7)
                    for pc in parts["bodies"]:
                        pc.set_facecolor(fill_col)
                        pc.set_alpha(0.35)
                        pc.set_linewidth(0)
                    parts["cmedians"].set_color(INK2)
                    parts["cmedians"].set_linewidth(1.2)
                    parts["cmedians"].set_zorder(4)
                except Exception:
                    pass

            jitter = rng.uniform(-0.18, 0.18, len(actions))
            ax.scatter(actions, i + jitter, s=7, alpha=0.45,
                       color=fill_col if cond not in ("plain", "plain_tom") else INK2,
                       linewidth=0, zorder=3)

        ax.set_title(cond, fontsize=9, fontweight="bold",
                     color=fill_col if cond != "plain" else INK)
        ax.set_xlabel("action (0–50)", fontsize=8, color=INK2)
        ax.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(col == 0)

        if col == 0:
            ax.set_yticks(range(n_cases))
            ax.set_yticklabels(case_order, fontsize=6)
            ax.set_ylabel("case  (sorted by original action)", fontsize=8, color=INK2)

    fig.suptitle("Action distribution per case × condition  "
                 "(plain · plain+ToM · 3 highest / 2 lowest Δ personas)\n"
                 "each violin = 20 model samples for that game state",
                 fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out / "7_violin_scatter.png")
    plt.close(fig)
    print(f"  saved 7_violin_scatter.png  ({n_cases} cases × {n_conds} conditions)")


# ── plot 8: behavioral fingerprint (characterise strategies) ──────────────────

def plot_behavioral_fingerprint(df: pd.DataFrame, out: Path) -> None:
    """2D: mean_delta (directional bias) vs within_case_std (model stochasticity).

    x-axis: does this persona systematically raise or lower the guess?
    y-axis: how stochastic is the model under this persona? (avg std across
            20 reps of the same prompt — meaningful even for plain/plain_tom,
            unlike between-case std of deltas which is zero for plain by construction)

    Quadrants:
      top-right  → raises guess AND is more stochastic
      top-left   → lowers guess AND is more stochastic
      bottom     → deterministic-ish (low within-case variance)
    """
    profile = cond_stats(df)

    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in profile.iterrows():
        col    = DIV_POS if row["mean_delta"] > 0 else DIV_NEG
        if row["condition"] in ("plain", "plain_tom"):
            col = MUTED
        marker = "D" if row["is_tom"] else "o"
        alpha  = 0.55 if row["is_tom"] else 0.85
        ec     = INK if row["condition"] in ("plain", "plain_tom") else "none"
        ax.scatter(row["mean_delta"], row["within_case_std"],
                   color=col, marker=marker, s=50, alpha=alpha,
                   linewidth=0.8, edgecolors=ec, zorder=3)
        if not row["is_tom"]:
            ax.text(row["mean_delta"] + 0.06, row["within_case_std"],
                    row["base_name"], fontsize=6.5, va="center",
                    color=INK2, alpha=0.9)

    ax.axvline(0, color=BASELINE, linewidth=1, linestyle="--")
    ax.set_xlabel("Mean Δ action  (← lowers guess · raises guess →)", color=INK2)
    ax.set_ylabel("Within-case std  (avg std of 20 reps on same prompt — model stochasticity)",
                  color=INK2)
    ax.set_title("Behavioral fingerprint of each persona\n"
                 "circle = base  ·  diamond = +ToM  ·  "
                 "grey = plain baselines  ·  blue = lowers  ·  red = raises",
                 fontsize=11, fontweight="bold", pad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out / "8_behavioral_fingerprint.png")
    plt.close(fig)
    print(f"  saved 8_behavioral_fingerprint.png  ({len(profile)} conditions)")


# ── plot 9: strategy similarity heatmap ───────────────────────────────────────

def plot_similarity_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Pearson correlation of per-case delta profiles, clustered.

    Two conditions are 'similar' if they push the agent's action in the same
    direction across the 50 game states (regardless of magnitude).
    """
    pivot = (df.groupby(["condition", "case_id"])["delta"]
               .mean()
               .unstack())                          # shape: conditions × cases
    pivot = pivot.dropna(thresh=int(0.6 * pivot.shape[1]))
    corr  = pivot.fillna(0).T.corr()               # conditions × conditions

    # hierarchical clustering for row/col order
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        dist = np.clip((1 - corr.values + (1 - corr.values).T) / 2, 0, None)
        np.fill_diagonal(dist, 0)
        link  = linkage(squareform(dist), method="average")
        order = leaves_list(link)
    except ImportError:
        order = np.argsort(corr.values.sum(axis=1))   # fallback: sort by row sum

    labels  = corr.index.tolist()
    ordered = [labels[i] for i in order]
    corr_o  = corr.iloc[order, :].iloc[:, order]

    n   = len(ordered)
    sz  = max(8, n * 0.22)
    fig, ax = plt.subplots(figsize=(sz, sz))

    # center the diverging colormap at the mean off-diagonal correlation,
    # not at 0 — if most personas correlate positively the midpoint shifts
    # accordingly so the palette contrast reveals structure, not the mean.
    vals_offdiag = corr_o.values[~np.eye(n, dtype=bool)]
    vcenter = float(np.mean(vals_offdiag))
    vext    = max(abs(vals_offdiag.max() - vcenter), abs(vcenter - vals_offdiag.min())) * 1.05
    norm    = mcolors.TwoSlopeNorm(vcenter=vcenter,
                                   vmin=vcenter - vext, vmax=vcenter + vext)
    im = ax.imshow(corr_o.values, cmap=diverging_cmap(),
                   norm=norm, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02,
                 label=f"Pearson r  (midpoint = mean r {vcenter:+.2f})")

    ax.set_xticks(range(n))
    ax.set_xticklabels(ordered, rotation=55, ha="right", fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ordered, fontsize=6)
    ax.set_title("Strategy similarity\n"
                 "(Pearson r of per-case Δ profiles · hierarchically clustered · "
                 "red = same direction, blue = opposite)",
                 fontsize=11, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(out / "9_strategy_similarity.png")
    plt.close(fig)
    print(f"  saved 9_strategy_similarity.png  ({n}×{n})")


# ── plot 10: case × condition heatmap — alphabetically sorted ────────────────

def plot_heatmap_alpha(df: pd.DataFrame, out: Path) -> None:
    """Same as plot 5 but conditions sorted alphabetically for easy lookup."""
    cond_order = sorted(df["condition"].unique())
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

    _shade_baseline_cols(ax, cond_order)
    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels(cond_order, rotation=60, ha="right", fontsize=7)
    _bold_baseline_ticks(ax, cond_order, axis="x")
    ax.set_yticks(range(len(case_order)))
    ax.set_yticklabels(case_order, fontsize=7)
    ax.set_title("Mean Δ action per case × condition  (conditions alphabetical)\n"
                 "(blue = guesses lower, red = higher than plain baseline)",
                 fontsize=11, fontweight="bold", pad=10)

    cat_seq  = df.set_index("case_id")["category"].to_dict()
    prev_cat = None
    for i, cid in enumerate(case_order):
        cat = cat_seq.get(cid, "")
        if cat != prev_cat and i > 0:
            ax.axhline(i - 0.5, color=INK, linewidth=0.6, alpha=0.4)
        prev_cat = cat

    fig.tight_layout()
    fig.savefig(out / "10_case_condition_heatmap_alpha.png")
    plt.close(fig)
    print(f"  saved 10_case_condition_heatmap_alpha.png  "
          f"({len(case_order)}×{len(cond_order)})")


# ── plot 11: ToM impact — does theory-of-mind shift strategies? ───────────────

def plot_tom_impact(df: pd.DataFrame, out: Path) -> None:
    """Two scatter panels comparing base vs +ToM for every persona.

    Left:  mean_delta(base) vs mean_delta(+ToM)   — directional shift
    Right: within_case_std(base) vs within_case_std(+ToM) — stochasticity shift

    Points on the diagonal = ToM has no effect.
    Points above = ToM pushes that metric higher.
    Points below = ToM pulls it lower.
    Includes plain vs plain_tom as a reference pair.
    """
    stats = cond_stats(df)
    base  = stats[~stats["is_tom"]].set_index("base_name")
    tom   = stats[ stats["is_tom"]].set_index("base_name")
    shared = sorted(set(base.index) & set(tom.index))
    if not shared:
        print("  skipped 10_tom_impact.png (no ToM pairs)")
        return

    b_delta  = base.loc[shared, "mean_delta"]
    t_delta  = tom.loc[shared,  "mean_delta"]
    b_std    = base.loc[shared, "within_case_std"]
    t_std    = tom.loc[shared,  "within_case_std"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, bvals, tvals, xlabel, ylabel, title in [
        (axes[0], b_delta, t_delta,
         "mean Δ action  (base persona)",
         "mean Δ action  (+ToM)",
         "Does ToM shift the directional bias?"),
        (axes[1], b_std, t_std,
         "within-case std  (base persona)",
         "within-case std  (+ToM)",
         "Does ToM change model stochasticity?"),
    ]:
        lo = min(bvals.min(), tvals.min()) - 0.3
        hi = max(bvals.max(), tvals.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1,
                linestyle="--", zorder=1, label="no ToM effect")

        for name in shared:
            bv, tv = bvals[name], tvals[name]
            shift  = tv - bv
            col    = DIV_POS if shift > 0 else (DIV_NEG if shift < 0 else MUTED)
            is_baseline = name in ("plain",)
            ax.annotate("", xy=(tv, tv), xytext=(bv, bv),   # invisible arrow placeholder
                        arrowprops=dict(arrowstyle="->", color=col, lw=0))
            ax.plot([bv, tv], [bv, tv], color=col, linewidth=0)  # suppress
            ax.scatter(bv, tv,
                       color=col, s=55 if not is_baseline else 90,
                       marker="s" if is_baseline else "o",
                       alpha=0.85, linewidth=0.6,
                       edgecolors=INK if is_baseline else "none",
                       zorder=3)
            ax.text(bv + 0.04, tv + 0.04, name, fontsize=6.5,
                    color=INK2, va="bottom", alpha=0.9)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(xlabel, color=INK2)
        ax.set_ylabel(ylabel, color=INK2)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.xaxis.grid(True, linestyle="--", alpha=0.35)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.set_aspect("equal")

    fig.suptitle("Theory-of-Mind impact on persona strategies\n"
                 "(square = plain baseline  ·  circle = persona  ·  "
                 "above diagonal = ToM raises  ·  below = ToM lowers)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "11_tom_impact.png")
    plt.close(fig)
    print(f"  saved 11_tom_impact.png  ({len(shared)} pairs)")


# ── plot 12: statistical test — which personas significantly shift behaviour? ─

def _bh_correct(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    n   = len(pvalues)
    idx = np.argsort(pvalues)
    sp  = pvalues[idx]
    q   = np.minimum(1.0, sp * n / (np.arange(1, n + 1)))
    for i in range(n - 2, -1, -1):          # enforce monotonicity
        q[i] = min(q[i], q[i + 1])
    out       = np.empty(n)
    out[idx]  = q
    return out


def _sig_label(q: float) -> str:
    if q < 0.001: return "***"
    if q < 0.01:  return "**"
    if q < 0.05:  return "*"
    return "ns"


def plot_statistical_test(df: pd.DataFrame, out: Path) -> None:
    """Wilcoxon signed-rank test per condition on 50 per-case mean deltas.

    H0: the persona has no systematic effect on the action (median Δ = 0 across
    game states).  Adjusted for multiple comparisons via Benjamini-Hochberg FDR.

    plain is excluded (its deltas are 0 by construction — it IS the baseline).
    plain_tom is included and tested.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        print("  skipped 12_statistical_test.png (scipy not available)")
        return

    per_case = (df.groupby(["condition", "case_id"])["delta"]
                  .mean()
                  .reset_index())

    test_conds = sorted([c for c in df["condition"].unique() if c != "plain"])
    rows = []
    for cond in test_conds:
        vals = per_case[per_case["condition"] == cond]["delta"].dropna().values
        if len(vals) < 5:
            continue
        try:
            _, pval = wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
        except Exception:
            # fallback: if all values identical wilcoxon raises ValueError
            from scipy.stats import ttest_1samp
            _, pval = ttest_1samp(vals, 0)
        rows.append({
            "condition":  cond,
            "mean_delta": float(np.mean(vals)),
            "median_delta": float(np.median(vals)),
            "pval":       float(pval),
            "n":          len(vals),
            "is_baseline": cond in BASELINES,
        })

    res = pd.DataFrame(rows)
    res["qval"]       = _bh_correct(res["pval"].values)
    res["neg_log10_q"]= -np.log10(np.clip(res["qval"].values, 1e-300, 1))
    res["sig"]        = res["qval"] < 0.05
    res["sig_label"]  = res["qval"].map(_sig_label)

    n_sig = res["sig"].sum()
    print(f"  {n_sig}/{len(res)} conditions significant at FDR 5%")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax_v, ax_b) = plt.subplots(
        1, 2, figsize=(16, max(7, len(res) * 0.22)),
        gridspec_kw={"width_ratios": [1, 1.4]})

    # ── left: volcano ─────────────────────────────────────────────────────────
    thresh_y = -np.log10(0.05)
    ax_v.axhline(thresh_y, color=MUTED, linewidth=0.9,
                 linestyle="--", label="FDR 5%")
    ax_v.axvline(0, color=BASELINE, linewidth=0.9, linestyle="--")

    for _, row in res.iterrows():
        if not row["sig"]:
            col = MUTED; alpha = 0.45; zorder = 1
        else:
            col = DIV_POS if row["mean_delta"] > 0 else DIV_NEG
            alpha = 0.9; zorder = 3
        marker = "s" if row["is_baseline"] else "o"
        ec     = INK if row["is_baseline"] else "none"
        ax_v.scatter(row["mean_delta"], row["neg_log10_q"],
                     color=col, alpha=alpha, s=55, marker=marker,
                     edgecolors=ec, linewidth=0.7, zorder=zorder)
        if row["sig"]:
            ax_v.text(row["mean_delta"] + 0.04, row["neg_log10_q"] + 0.05,
                      row["condition"], fontsize=6.5, color=INK2,
                      va="bottom", alpha=0.9)

    ax_v.set_xlabel("Mean Δ action  (per-case mean, relative to plain baseline)",
                    color=INK2)
    ax_v.set_ylabel("−log₁₀(q)  after Benjamini-Hochberg correction", color=INK2)
    ax_v.set_title(f"Volcano plot\n{n_sig}/{len(res)} conditions significant (FDR 5%)",
                   fontsize=10, fontweight="bold", pad=8)
    ax_v.legend(frameon=False, fontsize=8)
    ax_v.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax_v.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax_v.set_axisbelow(True)

    # ── right: sorted bar chart ───────────────────────────────────────────────
    res_sorted = res.sort_values("mean_delta").reset_index(drop=True)
    n = len(res_sorted)

    labels = res_sorted["condition"].tolist()
    _shade_baseline_bars(ax_b, labels)

    for i, row in res_sorted.iterrows():
        if not row["sig"]:
            col = MUTED; alpha = 0.25
        else:
            col = DIV_POS if row["mean_delta"] > 0 else DIV_NEG
            alpha = 0.85
        ax_b.barh(i, row["mean_delta"], height=0.7,
                  color=col, alpha=alpha, linewidth=0)

        # significance label at bar tip
        x_off = 0.12 if row["mean_delta"] >= 0 else -0.12
        ha    = "left" if row["mean_delta"] >= 0 else "right"
        ax_b.text(row["mean_delta"] + x_off, i,
                  row["sig_label"], va="center", ha=ha,
                  fontsize=8,
                  color=INK if row["sig"] else MUTED,
                  fontweight="bold" if row["sig"] else "normal")

    ax_b.axvline(0, color=BASELINE, linewidth=1)
    ax_b.set_yticks(range(n))
    ax_b.set_yticklabels(labels, fontsize=7.5)
    _bold_baseline_ticks(ax_b, labels)
    ax_b.set_xlabel("Mean Δ action  (sorted by effect size)", color=INK2)
    ax_b.set_title("Effect size × significance\n"
                   "(solid = FDR < 5% · faded = not significant · "
                   "* p<.05  ** p<.01  *** p<.001)",
                   fontsize=10, fontweight="bold", pad=8)
    ax_b.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax_b.set_axisbelow(True)

    fig.suptitle("Do personas significantly shift agent behaviour?\n"
                 "Wilcoxon signed-rank on 50 per-case mean deltas · "
                 "Benjamini-Hochberg FDR correction",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "12_statistical_test.png")
    plt.close(fig)
    print(f"  saved 12_statistical_test.png")


# ── plot 13: per-case significance heatmap ────────────────────────────────────

def plot_per_case_test(df: pd.DataFrame, out: Path) -> None:
    """For each of the 50 cases: full pairwise Mann-Whitney U between every persona pair.

    Layout: 1:2 landscape slide.
      Left 50%  — persona × persona significance heatmap.
                  Significant pairs: full diverging colour.
                  Non-significant:   flat warm grey (no fading — unreadable).
      Right 50% — game state text (metadata + system prompt + user prompt).

    BH FDR correction within each case independently.
    Saved to per_case/ (50 files).
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        print("  skipped per_case/ (scipy not available)")
        return

    sub = out / "per_case"
    sub.mkdir(parents=True, exist_ok=True)

    # load case prompts for the text panel
    cases_path = Path("cases/gbs/persona_impact_cases.json")
    case_data: dict = {}
    if cases_path.exists():
        for c in json.loads(cases_path.read_text(encoding="utf-8")):
            case_data[c["case_id"]] = c

    cond_order = sorted(df["condition"].unique())
    n_conds    = len(cond_order)
    n_pairs    = n_conds * (n_conds - 1) // 2
    cmap       = diverging_cmap()
    NONSIG     = [0.86, 0.85, 0.83, 1.0]   # warm flat grey for non-significant
    DIAG       = [0.94, 0.94, 0.93, 1.0]   # lighter for diagonal

    grouped = {(cid, cond): grp["action"].dropna().values
               for (cid, cond), grp in df.groupby(["case_id", "condition"])}

    cases = sorted(df["case_id"].unique())
    for k, case_id in enumerate(cases):
        cat  = df.loc[df["case_id"] == case_id, "category"].iloc[0]
        info = case_data.get(case_id, {})

        samp = {c: grouped.get((case_id, c), np.array([])) for c in cond_order}

        # ── pairwise tests ────────────────────────────────────────────────────
        pairs, pvals = [], []
        diff_mat = np.zeros((n_conds, n_conds))
        for i in range(n_conds):
            for j in range(i + 1, n_conds):
                a, b = samp[cond_order[i]], samp[cond_order[j]]
                if len(a) >= 3 and len(b) >= 3:
                    try:
                        _, pval = mannwhitneyu(a, b, alternative="two-sided")
                    except Exception:
                        pval = 1.0
                else:
                    pval = np.nan
                d = float(np.mean(a) - np.mean(b)) if len(a) and len(b) else 0.0
                pairs.append((i, j))
                pvals.append(pval)
                diff_mat[i, j] =  d
                diff_mat[j, i] = -d

        pval_arr = np.array(pvals, dtype=float)
        valid_m  = ~np.isnan(pval_arr)
        qval_arr = np.ones(len(pvals))
        if valid_m.any():
            qval_arr[valid_m] = _bh_correct(pval_arr[valid_m])

        BH_ALPHA = 0.01
        sig_mat = np.zeros((n_conds, n_conds), dtype=bool)
        for (i, j), q in zip(pairs, qval_arr):
            if q < BH_ALPHA:
                sig_mat[i, j] = sig_mat[j, i] = True

        n_sig = int(sig_mat.sum()) // 2

        # ── RGBA: significant = full colour, non-sig = flat grey ──────────────
        vmax = np.max(np.abs(diff_mat)) or 1
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

        rgba = np.ones((n_conds, n_conds, 4))
        for i in range(n_conds):
            for j in range(n_conds):
                if i == j:
                    rgba[i, j] = DIAG
                elif sig_mat[i, j]:
                    rgba[i, j] = [*cmap(norm(diff_mat[i, j]))[:3], 1.0]
                else:
                    rgba[i, j] = NONSIG

        # ── figure: 1:2 landscape, matrix left / text right ──────────────────
        mat_side = max(10, n_conds * 0.18)
        fig = plt.figure(figsize=(mat_side * 2, mat_side))
        gs  = fig.add_gridspec(1, 2, wspace=0.06)
        ax_mat = fig.add_subplot(gs[0])
        ax_txt = fig.add_subplot(gs[1])

        # ── matrix ────────────────────────────────────────────────────────────
        ax_mat.imshow(rgba, aspect="auto", interpolation="nearest",
                      extent=[-0.5, n_conds - 0.5, n_conds - 0.5, -0.5])
        _shade_baseline_cols(ax_mat, cond_order)

        ax_mat.set_xticks(range(n_conds))
        ax_mat.set_xticklabels(cond_order, rotation=55, ha="right", fontsize=5.5)
        _bold_baseline_ticks(ax_mat, cond_order, axis="x")
        ax_mat.set_yticks(range(n_conds))
        ax_mat.set_yticklabels(cond_order, fontsize=5.5)
        _bold_baseline_ticks(ax_mat, cond_order, axis="y")

        fp_naive = int(round(BH_ALPHA * n_pairs))
        fp_bh    = int(np.ceil(BH_ALPHA * n_sig)) if n_sig > 0 else 0

        ax_mat.set_title(
            "Which personas produce significantly different guesses\n"
            "from each other on this specific game state?\n"
            f"Colour = mean(row) − mean(col)  ·  Grey = not significant (BH-corrected)\n"
            f"{n_sig} / {n_pairs} pairs significant  "
            f"(Mann-Whitney U + Benjamini-Hochberg FDR, α = 1%)\n"
            f"Without correction: ~{fp_naive} false positives expected  ·  "
            f"With BH: ≤{fp_bh} false discoveries among {n_sig} found",
            fontsize=8, fontweight="bold", pad=8, loc="left")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_mat, shrink=0.4, pad=0.02,
                     label="mean action diff (row − col)")

        # ── text panel ────────────────────────────────────────────────────────
        ax_txt.axis("off")
        W = 52   # chars per wrapped line

        sys_p = info.get("system_prompt", "(system prompt not found)")
        usr_p = info.get("user_prompt",   "(user prompt not found)")

        # keep only the most recent conversation context for long histories
        if len(usr_p) > 1400:
            usr_p = "[... earlier rounds omitted ...]\n\n" + usr_p[-1200:]

        def _wrap_preserving_newlines(text: str, width: int) -> str:
            out = []
            for para in text.split("\n"):
                if len(para) > width:
                    out.extend(textwrap.wrap(para, width=width) or [""])
                else:
                    out.append(para)
            return "\n".join(out)

        def section(header: str, body: str, width: int = W) -> str:
            bar = "-" * width
            wrapped = _wrap_preserving_newlines(body, width=width)
            return f"{bar}\n{header}\n{bar}\n{wrapped}"

        content = "\n\n".join([
            section("SYSTEM PROMPT", sys_p),
            section("USER PROMPT (conversation history)", usr_p),
        ])

        # cap to what fits at this font size
        content_lines = content.split("\n")
        content = "\n".join(content_lines[:55])

        ax_txt.text(0.0, 1.0, content,
                    transform=ax_txt.transAxes,
                    fontsize=11, va="top", ha="left",
                    fontfamily="monospace", linespacing=1.4,
                    clip_on=True)

        # ── suptitle: case identity ────────────────────────────────────────────
        rnd    = info.get("round", "?")
        tgt    = info.get("target", "?")
        orig   = info.get("original_action", "?")
        fig.suptitle(
            f"Case {case_id}  ·  {cat.replace('_', ' ').title()}  ·  "
            f"Round {rnd}  ·  Target {tgt}  ·  Original action: {orig}",
            fontsize=10, fontweight="bold", y=1.005)

        fig.savefig(sub / f"case_{case_id}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{k+1:02d}/{len(cases)}] case_{case_id}.png  ({n_sig}/{n_pairs} sig)")

    print(f"  saved {len(cases)} slides -> {sub}/")


# ── plot 14: per-category pairwise significance ───────────────────────────────

def plot_category_pairwise_test(df: pd.DataFrame, out: Path) -> None:
    """One 1:2 slide per category: pairwise Mann-Whitney U on *delta* values,
    pooled across all cases in the category.  Right panel summarises the
    category, lists cases, and calls out the most-divergent pairs.
    Saved to per_category/ (6 files).
    """
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        print("  skipped per_category/ (scipy not available)")
        return

    BH_ALPHA   = 0.01
    cond_order = sorted(df["condition"].unique())
    n_conds    = len(cond_order)
    n_pairs    = n_conds * (n_conds - 1) // 2
    cmap       = diverging_cmap()
    NONSIG     = [0.86, 0.85, 0.83, 1.0]
    DIAG       = [0.94, 0.94, 0.93, 1.0]

    # load case metadata for the text panel
    cases_path = Path("cases/gbs/persona_impact_cases.json")
    case_meta: dict = {}
    if cases_path.exists():
        for c in json.loads(cases_path.read_text(encoding="utf-8")):
            case_meta[c["case_id"]] = c

    sub = out / "per_category"
    sub.mkdir(parents=True, exist_ok=True)

    cats_present = [c for c in CAT_ORDER if c in df["category"].unique()]

    for cat in cats_present:
        color    = CAT_COLORS.get(cat, MUTED)
        cat_df   = df[df["category"] == cat]
        n_cases  = cat_df["case_id"].nunique()
        cat_cases = sorted(cat_df["case_id"].unique())
        n_samp   = n_cases * 20   # approximate

        # ── pairwise tests on raw actions ────────────────────────────────────
        samp = {c: cat_df[cat_df["condition"] == c]["action"].dropna().values
                for c in cond_order}

        pairs, pvals = [], []
        diff_mat = np.zeros((n_conds, n_conds))
        for i in range(n_conds):
            for j in range(i + 1, n_conds):
                a, b = samp[cond_order[i]], samp[cond_order[j]]
                if len(a) >= 3 and len(b) >= 3:
                    try:
                        _, pval = mannwhitneyu(a, b, alternative="two-sided")
                    except Exception:
                        pval = 1.0
                else:
                    pval = np.nan
                d = float(np.mean(a) - np.mean(b)) if len(a) and len(b) else 0.0
                pairs.append((i, j))
                pvals.append(pval)
                diff_mat[i, j] =  d
                diff_mat[j, i] = -d

        pval_arr = np.array(pvals, dtype=float)
        valid_m  = ~np.isnan(pval_arr)
        qval_arr = np.ones(len(pvals))
        if valid_m.any():
            qval_arr[valid_m] = _bh_correct(pval_arr[valid_m])

        sig_mat = np.zeros((n_conds, n_conds), dtype=bool)
        for (i, j), q in zip(pairs, qval_arr):
            if q < BH_ALPHA:
                sig_mat[i, j] = sig_mat[j, i] = True

        n_sig    = int(sig_mat.sum()) // 2
        fp_naive = int(round(BH_ALPHA * n_pairs))
        fp_bh    = int(np.ceil(BH_ALPHA * n_sig)) if n_sig > 0 else 0

        # collect top significant pairs by |mean delta diff|
        sig_pairs_ranked = []
        for (i, j), q in sorted(zip(pairs, qval_arr), key=lambda x: x[1]):
            if q < BH_ALPHA:
                ci, cj = cond_order[i], cond_order[j]
                d = diff_mat[i, j]
                sig_pairs_ranked.append((ci, cj, d, q))
        top_pairs = sig_pairs_ranked[:12]

        # ── RGBA ─────────────────────────────────────────────────────────────
        vmax = np.max(np.abs(diff_mat)) or 1
        norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

        rgba = np.ones((n_conds, n_conds, 4))
        for i in range(n_conds):
            for j in range(n_conds):
                if i == j:
                    rgba[i, j] = DIAG
                elif sig_mat[i, j]:
                    rgba[i, j] = [*cmap(norm(diff_mat[i, j]))[:3], 1.0]
                else:
                    rgba[i, j] = NONSIG

        # ── figure ───────────────────────────────────────────────────────────
        mat_side = max(10, n_conds * 0.18)
        fig = plt.figure(figsize=(mat_side * 2, mat_side))
        fig.patch.set_facecolor(SURFACE)
        gs  = fig.add_gridspec(1, 2, wspace=0.06)
        ax_mat = fig.add_subplot(gs[0])
        ax_txt = fig.add_subplot(gs[1])

        # ── heatmap ──────────────────────────────────────────────────────────
        ax_mat.imshow(rgba, aspect="auto", interpolation="nearest",
                      extent=[-0.5, n_conds - 0.5, n_conds - 0.5, -0.5])
        _shade_baseline_cols(ax_mat, cond_order)

        ax_mat.set_xticks(range(n_conds))
        ax_mat.set_xticklabels(cond_order, rotation=55, ha="right", fontsize=5.5)
        _bold_baseline_ticks(ax_mat, cond_order, axis="x")
        ax_mat.set_yticks(range(n_conds))
        ax_mat.set_yticklabels(cond_order, fontsize=5.5)
        _bold_baseline_ticks(ax_mat, cond_order, axis="y")

        for spine in ax_mat.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

        ax_mat.set_title(
            "Do strategies differ significantly within this category?\n"
            "Colour = mean action diff (row - col)   Grey = not significant (BH-corrected)\n"
            f"{n_sig} / {n_pairs} pairs significant  "
            f"(Mann-Whitney U + Benjamini-Hochberg FDR, alpha = 1%)\n"
            f"Without correction: ~{fp_naive} false positives expected  .  "
            f"With BH: <= {fp_bh} false discoveries among {n_sig} found",
            fontsize=8, fontweight="bold", pad=8, loc="left")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_mat, shrink=0.4, pad=0.02,
                     label="mean action diff (row - col)")

        # ── text panel ───────────────────────────────────────────────────────
        ax_txt.axis("off")
        cat_label = cat.replace("_", " ").title()
        desc      = CAT_META.get(cat, {}).get("desc", "")

        lines = []

        # category header
        lines += [
            "=" * 52,
            f"CATEGORY: {cat_label.upper()}",
            f"  {desc}",
            "=" * 52,
            "",
            f"  {n_cases} cases  |  ~{n_samp} samples per persona",
            f"  {n_sig}/{n_pairs} pairs significant (BH alpha=1%)",
            f"  Without correction: ~{fp_naive} false positives expected",
            f"  With BH guarantee:  <= {fp_bh} false discoveries among {n_sig} found",
            "",
        ]

        # cases list
        lines += ["CASES IN THIS CATEGORY", "-" * 52]
        for cid in cat_cases:
            m = case_meta.get(cid, {})
            rnd  = m.get("round", "?")
            tgt  = m.get("target", "?")
            orig = m.get("original_action", "?")
            lines.append(f"  {cid:<18}  rnd {rnd:>2}  tgt {tgt:>4}  "
                         f"orig action {orig:>2}")
        lines += [""]

        # top significantly different pairs
        if top_pairs:
            lines += ["TOP SIGNIFICANTLY DIFFERENT PAIRS", "-" * 52]
            for ci, cj, d, q in top_pairs:
                sign  = "+" if d > 0 else ""
                lines.append(f"  {ci:<22} vs {cj:<22}  "
                              f"d={sign}{d:+.1f}  q={q:.4f}")
        else:
            lines += ["No significant pairs found at alpha=1%."]

        lines += ["", "-" * 52,
                  "Raw action values compared directly.",
                  "plain is one condition among 70 -- no special baseline role."]

        # cap to visible area
        content = "\n".join(lines[:60])

        ax_txt.text(0.02, 0.98, content,
                    transform=ax_txt.transAxes,
                    fontsize=10, va="top", ha="left",
                    fontfamily="monospace", linespacing=1.45,
                    clip_on=True)

        # ── suptitle ─────────────────────────────────────────────────────────
        fig.suptitle(
            f"{cat_label}  --  Pairwise Persona Significance  "
            f"({n_cases} cases, ~{n_samp} samples/persona)",
            fontsize=11, fontweight="bold", y=1.005, color=color)

        fig.savefig(sub / f"cat_{cat}.png", dpi=110, bbox_inches="tight",
                    facecolor=SURFACE)
        plt.close(fig)
        print(f"  cat_{cat}.png  ({n_sig}/{n_pairs} sig)")

    print(f"  saved {len(cats_present)} category slides -> {sub}/")


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

    plot_case_taxonomy(args.out)
    plot_mean_delta(df, args.out)
    plot_variance(df, args.out)
    plot_tom_dumbbell(df, args.out)
    plot_category_breakdown(df, args.out)
    plot_heatmap(df, args.out)
    plot_oscillation_focus(df, args.out)
    plot_violin_scatter(df, args.out)
    plot_behavioral_fingerprint(df, args.out)
    plot_similarity_heatmap(df, args.out)
    plot_heatmap_alpha(df, args.out)
    plot_tom_impact(df, args.out)
    plot_statistical_test(df, args.out)
    plot_per_case_test(df, args.out)
    plot_category_pairwise_test(df, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
