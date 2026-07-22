"""Plot results of the Picking / Persona sweep (Riedl 2025, arXiv 2510.05174).

Reads every episode summary under logs/picking_sweep/<run_id>/ and produces:

  figures/picking_sweep/success_rate.png        — fraction of episodes that converged,
                                                   per condition × player count × model
                                                   (replicates paper Figure 2a)
  figures/picking_sweep/rounds_to_success.png   — mean rounds conditional on convergence
                                                   (replicates paper Appendix A1)
  figures/picking_sweep/n_datapoints.png        — episodes collected so far

Usage
-----
python scripts/plot_picking_sweep.py
"""
from __future__ import annotations

import csv
import json
import re
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LOGS_DIR = Path("logs/picking_sweep")
OUT_CSV  = Path("picking_sweep_summary.csv")
FIG_DIR  = Path("figures/picking_sweep")

RUN_ID_RE = re.compile(
    r"^gbs_exact_replication_(?P<condition>plain|persona|tom)_(?P<players>\d+)p(?P<model>_20b|_14b)?$"
)

CONDITION_ORDER = ["plain", "persona", "tom"]
CONDITION_LABEL = {"plain": "plain", "persona": "persona", "tom": "ToM"}
PLAYERS_ORDER   = [2, 3, 10]
MODEL_ORDER     = ["Qwen3-14B", "gpt-oss-20b"]

PLAYER_COLOR = {2: "#2a78d6", 3: "#1baf7a", 10: "#eda100"}
MUTED = "#898781"
GRID  = "#e1e0d9"


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_run_id(run_id: str) -> dict | None:
    m = RUN_ID_RE.match(run_id)
    if not m:
        return None
    return {
        "condition": m.group("condition"),
        "players":   int(m.group("players")),
        "model":     {"_20b": "gpt-oss-20b", "_14b": "Qwen3-14B"}.get(m.group("model"), "Qwen3-14B"),
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
                "run_id": run_dir.name,
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
                ax.text(x, y, lab, ha="center", va="bottom", fontsize=7, color="#52514e")


def add_player_legend(fig):
    handles = [Patch(facecolor=PLAYER_COLOR.get(n, MUTED), label=f"{n}p")
               for n in PLAYERS_ORDER]
    fig.legend(handles=handles, title="players", fontsize=8, title_fontsize=8,
               frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.97))


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_success_rate(rows: list[dict]):
    groups = group_rows(rows, ("model", "condition", "players"))

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(10, 4.2), sharey=True)
    fig.suptitle("Picking — success rate per condition (fraction of episodes converged)",
                 fontsize=11, color="#0b0b0b")

    def value_fn(grp):
        if not grp:
            return None, 0.0, ""
        n_conv = sum(1 for r in grp if r["converged"])
        frac   = n_conv / len(grp)
        return frac, 0.0, f"{n_conv}/{len(grp)}"

    for ax, model in zip(axes, MODEL_ORDER):
        bar_group(ax, groups, (model,), value_fn, label_fmt=True)
        style_axis(ax, model, "success rate" if model == MODEL_ORDER[0] else "")
        ax.set_ylim(0, 1.15)

    add_player_legend(fig)
    fig.text(0.5, 0.01, "Bar labels: converged / total episodes.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "success_rate.png", dpi=160)
    plt.close(fig)


def plot_rounds_to_success(rows: list[dict]):
    groups = group_rows(rows, ("model", "condition", "players"))

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(10, 4.2), sharey=True)
    fig.suptitle("Picking — mean rounds to convergence (successful episodes only)",
                 fontsize=11, color="#0b0b0b")

    def value_fn(grp):
        converged = [r["converged_round"] for r in grp
                     if r["converged"] and r["converged_round"] is not None]
        val, err = mean_sem(converged)
        label = str(len(converged)) if converged else ""
        return val, err, label

    for ax, model in zip(axes, MODEL_ORDER):
        bar_group(ax, groups, (model,), value_fn, label_fmt=True)
        style_axis(ax, model, "rounds to convergence" if model == MODEL_ORDER[0] else "")

    add_player_legend(fig)
    fig.text(0.5, 0.01, "Bar labels: number of converged episodes used.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "rounds_to_success.png", dpi=160)
    plt.close(fig)


def plot_n_datapoints(rows: list[dict]):
    groups = group_rows(rows, ("model", "condition", "players"))

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(10, 4.2), sharey=False)
    fig.suptitle("Picking — episodes collected so far (sweep in progress)",
                 fontsize=11, color="#0b0b0b")

    def value_fn(grp):
        return (len(grp), 0.0, str(len(grp))) if grp else (0, 0.0, "0")

    for ax, model in zip(axes, MODEL_ORDER):
        bar_group(ax, groups, (model,), value_fn, label_fmt=True)
        style_axis(ax, model, "n episodes" if model == MODEL_ORDER[0] else "")

    add_player_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "n_datapoints.png", dpi=160)
    plt.close(fig)


def write_csv(rows: list[dict]):
    fieldnames = ["run_id", "condition", "players", "model", "converged", "converged_round"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = collect_rows()
    if not rows:
        print(f"No episode summaries found under {LOGS_DIR}")
        return
    write_csv(rows)
    print(f"Wrote {len(rows)} episode rows -> {OUT_CSV}")
    plot_success_rate(rows)
    plot_rounds_to_success(rows)
    plot_n_datapoints(rows)
    print(f"Wrote figures -> {FIG_DIR}/")


if __name__ == "__main__":
    main()
