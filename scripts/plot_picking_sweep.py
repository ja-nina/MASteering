"""Plot results of the Picking / Persona sweep (Riedl 2025, arXiv 2510.05174).

Reads every episode summary under logs/picking_sweep/<run_id>/ and produces
separate figure sets per model under:

  figures/picking_sweep/qwen3_14b/
    success_rate.png       — convergence rate per condition × player count
    rounds_to_success.png  — mean rounds conditional on convergence
    n_datapoints.png       — episode counts
    convergence_line.png   — line chart: convergence rate vs group size
    box_10p.png            — round distributions at 10 players

  figures/picking_sweep/gpt_oss_20b/
    (same set; sparse plots are skipped where N < 10)

Usage
-----
python scripts/plot_picking_sweep.py
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics as stats
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LOGS_DIR = Path("logs/picking_sweep")
OUT_CSV  = Path("picking_sweep_summary.csv")
BASE_FIG = Path("figures/picking_sweep")

RUN_ID_RE = re.compile(
    r"^gbs_exact_replication_(?P<condition>plain|persona|tom)"
    r"_(?P<players>\d+)p(?P<model>_20b|_14b)?$"
)

CONDITION_ORDER = ["plain", "persona", "tom"]
CONDITION_LABEL = {"plain": "plain", "persona": "persona", "tom": "ToM"}
PLAYERS_ORDER   = [2, 3, 10]

PLAYER_COLOR = {2: "#2a78d6", 3: "#1baf7a", 10: "#eda100"}
COND_COLOR   = {"plain": "#2a78d6", "persona": "#eb6834", "tom": "#1baf7a"}
MUTED = "#898781"
GRID  = "#e1e0d9"

# minimum episodes to include a group in plots
MIN_N = 10
# non-converged episodes are treated as having taken this many rounds
CAP_ROUNDS = 30

MODELS = [
    ("Qwen3-14B",   "qwen3_14b"),
    ("gpt-oss-20b", "gpt_oss_20b"),
]


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_run_id(run_id: str) -> dict | None:
    m = RUN_ID_RE.match(run_id)
    if not m:
        return None
    return {
        "condition": m.group("condition"),
        "players":   int(m.group("players")),
        "model":     {"_20b": "gpt-oss-20b", "_14b": "Qwen3-14B"}.get(
                         m.group("model"), "Qwen3-14B"),
    }


def collect_rows() -> list[dict]:
    rows = []
    for run_dir in sorted(LOGS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_id(run_dir.name)
        if parsed is None:
            continue
        for summary_path in sorted(run_dir.glob("episode_*.summary.json")):
            try:
                with open(summary_path, encoding="utf-8") as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            rows.append({
                "run_id":          run_dir.name,
                **parsed,
                "converged":       summary.get("gbs_converged", False),
                "converged_round": summary.get("gbs_converged_round"),
            })
    return rows


# ── grouping + stats ──────────────────────────────────────────────────────────

def mean_sem(values: list[float]) -> tuple[float | None, float]:
    values = [v for v in values if v is not None]
    if not values:
        return None, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return stats.mean(values), stats.stdev(values) / len(values) ** 0.5


def wilson_ci(n: int, k: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    centre = (k + z * z / 2) / (n + z * z)
    half   = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return max(0.0, centre - half), min(1.0, centre + half)


def effective_rounds(r: dict) -> int:
    """Round count for a single episode; non-converged episodes are capped at CAP_ROUNDS."""
    if r["converged"] and r["converged_round"] is not None:
        return r["converged_round"]
    return CAP_ROUNDS


def group_rows(rows: list[dict], keys: tuple[str, ...]) -> dict:
    groups: dict = {}
    for r in rows:
        k = tuple(r[key] for key in keys)
        groups.setdefault(k, []).append(r)
    return groups


# ── shared axis styling ───────────────────────────────────────────────────────

def style_axis(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=10, color="#0b0b0b")
    ax.set_ylabel(ylabel, fontsize=9, color="#52514e")
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER], fontsize=9)
    ax.tick_params(axis="y", labelsize=8, colors="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def bar_group(ax, groups: dict, base_key: tuple, value_fn, label_fmt=False):
    n_players = len(PLAYERS_ORDER)
    width = 0.8 / n_players
    for i, n in enumerate(PLAYERS_ORDER):
        xs, ys, errs, labels = [], [], [], []
        for j, cond in enumerate(CONDITION_ORDER):
            key = base_key + (cond, n)
            grp = groups.get(key, [])
            if len(grp) < MIN_N:
                continue
            val, err, lab = value_fn(grp)
            if val is None:
                continue
            xs.append(j - 0.4 + width * (i + 0.5))
            ys.append(val)
            errs.append(err)
            labels.append(lab)
        if not xs:
            continue
        color = PLAYER_COLOR.get(n, MUTED)
        ax.bar(xs, ys, width=width * 0.9, yerr=errs, capsize=3,
               color=color, label=f"{n}p", zorder=3,
               error_kw={"ecolor": "#52514e", "elinewidth": 1})
        if label_fmt:
            for x, y, lab in zip(xs, ys, labels):
                ax.text(x, y, lab, ha="center", va="bottom",
                        fontsize=7, color="#52514e")


def add_player_legend(fig):
    handles = [Patch(facecolor=PLAYER_COLOR.get(n, MUTED), label=f"{n}p")
               for n in PLAYERS_ORDER]
    fig.legend(handles=handles, title="players", fontsize=8, title_fontsize=8,
               frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.97))


# ── plots (each takes model name + output directory) ─────────────────────────

def plot_success_rate(rows: list[dict], model: str, out_dir: Path):
    groups = group_rows(rows, ("model", "condition", "players"))

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    fig.suptitle(f"Picking — success rate per condition\n{model}",
                 fontsize=11, color="#0b0b0b")

    def value_fn(grp):
        if not grp:
            return None, 0.0, ""
        n_conv = sum(1 for r in grp if r["converged"])
        return n_conv / len(grp), 0.0, f"{n_conv}/{len(grp)}"

    bar_group(ax, groups, (model,), value_fn, label_fmt=True)
    style_axis(ax, model, "success rate")
    ax.set_ylim(0, 1.15)
    add_player_legend(fig)
    fig.text(0.5, 0.01, "Bar labels: converged / total episodes.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "success_rate.png", dpi=160)
    plt.close(fig)


def plot_rounds_to_success(rows: list[dict], model: str, out_dir: Path):
    groups = group_rows(rows, ("model", "condition", "players"))

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    fig.suptitle(f"Picking — mean rounds (non-converged capped at {CAP_ROUNDS})\n{model}",
                 fontsize=11, color="#0b0b0b")

    def value_fn(grp):
        all_rounds = [effective_rounds(r) for r in grp]
        val, err = mean_sem(all_rounds)
        return val, err, str(len(grp)) if all_rounds else ""

    bar_group(ax, groups, (model,), value_fn, label_fmt=True)
    style_axis(ax, model, f"mean rounds (cap={CAP_ROUNDS})")
    add_player_legend(fig)
    fig.text(0.5, 0.01,
             f"All episodes included. Non-converged treated as {CAP_ROUNDS} rounds.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "rounds_to_success.png", dpi=160)
    plt.close(fig)


def plot_n_datapoints(rows: list[dict], model: str, out_dir: Path):
    groups = group_rows(rows, ("model", "condition", "players"))

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    fig.suptitle(f"Picking — episodes collected\n{model}",
                 fontsize=11, color="#0b0b0b")

    def value_fn(grp):
        return (len(grp), 0.0, str(len(grp))) if grp else (None, 0.0, "")

    bar_group(ax, groups, (model,), value_fn, label_fmt=True)
    style_axis(ax, model, "n episodes")
    add_player_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "n_datapoints.png", dpi=160)
    plt.close(fig)


def plot_convergence_line(rows: list[dict], model: str, out_dir: Path):
    """Line chart: convergence rate vs group size, one line per condition."""
    groups = group_rows(rows, ("model", "condition", "players"))

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    xs = list(range(len(PLAYERS_ORDER)))
    any_plotted = False

    for cond in CONDITION_ORDER:
        ys, err_lo, err_hi, valid_xs = [], [], [], []
        for xi, n_players in enumerate(PLAYERS_ORDER):
            grp = groups.get((model, cond, n_players), [])
            if len(grp) < MIN_N:
                continue
            n_conv = sum(1 for r in grp if r["converged"])
            rate   = n_conv / len(grp)
            clo, chi = wilson_ci(len(grp), n_conv)
            valid_xs.append(xi)
            ys.append(rate)
            err_lo.append(rate - clo if rate < 1.0 else 0)
            err_hi.append(chi - rate if rate < 1.0 else 0)

        if not valid_xs:
            continue
        any_plotted = True
        color = COND_COLOR[cond]
        ax.errorbar(valid_xs, ys, yerr=[err_lo, err_hi],
                    color=color, linewidth=2, marker="o", markersize=7,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=1.5, label=CONDITION_LABEL[cond], zorder=3,
                    ecolor=color, capsize=4, capthick=1.3, elinewidth=1.3)
        if not math.isnan(ys[-1]):
            ax.annotate(f"{ys[-1]:.1%}", xy=(valid_xs[-1], ys[-1]),
                        xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=color, fontweight="bold")

    if not any_plotted:
        ax.text(0.5, 0.5, f"Insufficient data (N < {MIN_N}) for all groups",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=MUTED)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p}p" for p in PLAYERS_ORDER])
    ax.set_ylim(0.65, 1.06)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_yticks([v / 100 for v in range(65, 105, 5)])
    ax.set_ylabel("Convergence rate", fontsize=9, color="#52514e")
    ax.set_xlabel("Group size (players)", fontsize=9, color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8, colors="#52514e")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Convergence Rate vs Group Size", fontsize=11, color="#0b0b0b")
    ax.text(0.98, 0.04, f"{model} · error bars = 95% Wilson CI",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=MUTED)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "convergence_line.png", dpi=160)
    plt.close(fig)


def plot_box_10p(rows: list[dict], model: str, out_dir: Path):
    """Box plots of round distributions at 10 players; non-converged episodes capped at CAP_ROUNDS."""
    groups = group_rows(rows, ("model", "condition", "players"))

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    xs = list(range(len(CONDITION_ORDER)))
    any_plotted = False

    for i, cond in enumerate(CONDITION_ORDER):
        grp = groups.get((model, cond, 10), [])
        if len(grp) < MIN_N:
            ax.text(xs[i], 15, f"N={len(grp)}\n(too few)", ha="center", va="center",
                    fontsize=8, color=MUTED)
            continue
        rounds = [effective_rounds(r) for r in grp]
        any_plotted = True
        color = COND_COLOR[cond]
        ax.boxplot(rounds, positions=[xs[i]], widths=0.44,
                   patch_artist=True, showfliers=True,
                   medianprops=dict(color=color, linewidth=2.5),
                   boxprops=dict(facecolor=color + "2e", edgecolor=color, linewidth=1.5),
                   whiskerprops=dict(color=color, linewidth=1.3, linestyle="-"),
                   capprops=dict(color=color, linewidth=1.3),
                   flierprops=dict(marker="o", markersize=3,
                                   markerfacecolor=color, alpha=0.35,
                                   markeredgecolor="none"),
                   zorder=3)
        med = stats.median(rounds)
        n_conv = sum(1 for r in grp if r["converged"])
        ax.text(xs[i], med - 0.9, f"med={int(med)}", ha="center", va="top",
                fontsize=8, color="#52514e")
        ax.text(xs[i], -1.5, f"N={n_conv}/{len(grp)}", ha="center", va="top",
                fontsize=7.5, color=MUTED)

    ax.set_xticks(xs)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER])
    ax.set_ylim(-3, 33)
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_ylabel("Rounds to convergence", fontsize=9, color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8, colors="#52514e")
    ax.set_title("Round Distribution at 10 Players", fontsize=11, color="#0b0b0b")
    ax.text(0.98, 0.97, f"{model} · non-converged capped at {CAP_ROUNDS}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=MUTED)
    ax.text(0.5, -0.13, "N = converged / total episodes",
            transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "box_10p.png", dpi=160)
    plt.close(fig)


def plot_violin(rows: list[dict], model: str, out_dir: Path):
    """Violin + strip charts for all group sizes; non-converged episodes capped at CAP_ROUNDS."""
    groups = group_rows(rows, ("model", "condition", "players"))
    rng = random.Random(42)

    fig, axes = plt.subplots(1, len(PLAYERS_ORDER), figsize=(11, 4.8), sharey=True)
    fig.suptitle(
        f"Round distributions — all episodes (non-converged capped at {CAP_ROUNDS})\n{model}",
        fontsize=11, color="#0b0b0b",
    )

    for ax, n_players in zip(axes, PLAYERS_ORDER):
        ax.set_title(f"{n_players} players", fontsize=9, color="#52514e")

        for i, cond in enumerate(CONDITION_ORDER):
            grp = groups.get((model, cond, n_players), [])
            color = COND_COLOR[cond]

            if len(grp) < MIN_N:
                ax.text(i, 15, f"N={len(grp)}\n(too few)", ha="center", va="center",
                        fontsize=8, color=MUTED)
                continue

            rounds = [effective_rounds(r) for r in grp]
            n_conv = sum(1 for r in grp if r["converged"])

            # violin body
            parts = ax.violinplot([rounds], positions=[i], widths=0.65,
                                  showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color + "38")
                pc.set_edgecolor(color)
                pc.set_linewidth(1.5)
                pc.set_alpha(1.0)

            # jittered strip — individual episodes
            xs = [i + rng.uniform(-0.13, 0.13) for _ in rounds]
            ax.scatter(xs, rounds, color=color, s=7, alpha=0.35,
                       linewidths=0, zorder=4)

            # median bar (solid)
            med = stats.median(rounds)
            ax.hlines(med, i - 0.22, i + 0.22, color=color, linewidth=2.2, zorder=5)

            # mean marker (dashed line + diamond)
            mn = stats.mean(rounds)
            ax.hlines(mn, i - 0.22, i + 0.22, color=color, linewidth=1.5,
                      linestyle="--", zorder=5)
            ax.scatter([i], [mn], marker="D", s=28, color=color,
                       edgecolors="white", linewidths=0.8, zorder=6)

            # exact numbers: median and mean above, N below
            ax.text(i, CAP_ROUNDS + 0.8,
                    f"med={int(med)}  μ={mn:.1f}",
                    ha="center", va="bottom", fontsize=7.5, color="#52514e")
            ax.text(i, -2.2, f"{n_conv}/{len(grp)}", ha="center", va="top",
                    fontsize=7.5, color=MUTED)

        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors="#52514e")

    axes[0].set_ylim(-4, CAP_ROUNDS + 5)
    axes[0].set_yticks([0, 5, 10, 15, 20, 25, 30])
    axes[0].set_ylabel("Rounds", fontsize=9, color="#52514e")
    axes[0].tick_params(axis="y", labelsize=8, colors="#52514e")
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    fig.text(
        0.5, 0.01,
        f"N = converged / total  ·  dots = individual episodes"
        f"  ·  solid bar = median  ·  dashed bar + diamond = mean"
        f"  ·  non-converged counted as {CAP_ROUNDS} rounds",
        ha="center", fontsize=7.5, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "violin.png", dpi=160)
    plt.close(fig)


# ── CSV ───────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict]):
    fieldnames = ["run_id", "condition", "players", "model", "converged", "converged_round"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    rows = collect_rows()
    if not rows:
        print(f"No episode summaries found under {LOGS_DIR}")
        return

    write_csv(rows)
    print(f"Wrote {len(rows)} episode rows -> {OUT_CSV}\n")

    for model, slug in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        n_eps = len(model_rows)
        if n_eps == 0:
            print(f"[{model}] no episodes found — skipping")
            continue

        out_dir = BASE_FIG / slug
        print(f"[{model}] {n_eps} episodes -> {out_dir}/")

        plot_success_rate(model_rows, model, out_dir)
        plot_rounds_to_success(model_rows, model, out_dir)
        plot_n_datapoints(model_rows, model, out_dir)
        plot_convergence_line(model_rows, model, out_dir)
        plot_box_10p(model_rows, model, out_dir)
        plot_violin(model_rows, model, out_dir)

        print(f"  done: success_rate, rounds_to_success, n_datapoints,"
              f" convergence_line, box_10p, violin")

    print(f"\nAll figures in {BASE_FIG}/")


if __name__ == "__main__":
    main()
