"""Aggregate and plot the reasoning-mode sweep (noop / non_thinking / thinking).

Reads every episode summary + JSONL trace under
logs/reasoning_sweep/<game>_<mode>_<n>p[_20b]/ and produces:

  - reasoning_sweep_summary.csv                       (one row per episode)
  - figures/reasoning_sweep/rounds_to_success_gbs.png (GBS only — beauty_contest
    has no convergence criterion, it always plays a fixed number of rounds)
  - figures/reasoning_sweep/response_length.png       (mean completion length,
    both games)
  - figures/reasoning_sweep/n_datapoints.png          (episodes collected so
    far per condition — the sweep is still running, coverage is uneven)

Usage
-----
python scripts/plot_reasoning_sweep.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LOGS_DIR = Path("logs/reasoning_sweep")
OUT_CSV = Path("reasoning_sweep_summary.csv")
FIG_DIR = Path("figures/reasoning_sweep")

RUN_ID_RE = re.compile(r"^(?P<game>beauty_contest|gbs)_(?P<mode>non_thinking|noop_thinking|tom_thinking|noop|thinking|tom)_(?P<players>\d+)p(?P<model>_20b)?$")

# Modes plotted per model — Qwen3 only runs the 2x2 ToM ablation
# (noop/noop_thinking × no-ToM/ToM); step-by-step and thinking are 20b-only.
MODEL_MODES: dict[str, list[str]] = {
    "Qwen3-4B":   ["noop", "noop_thinking", "tom", "tom_thinking"],
    "gpt-oss-20b": ["noop", "noop_thinking", "non_thinking", "thinking", "tom", "tom_thinking"],
}
MODE_LABEL = {"noop": "noop", "noop_thinking": "noop+think",
              "non_thinking": "step-by-step", "thinking": "thinking",
              "tom": "tom", "tom_thinking": "tom+think"}

# Modes drawn as horizontal reference lines so tom/tom_thinking can be compared
# against the no-prompt baselines.
REFERENCE_MODES = {"noop": "--", "noop_thinking": ":"}
PLAYERS_ORDER = [2, 3, 4]
MODEL_ORDER = ["Qwen3-4B", "gpt-oss-20b"]
GAME_ORDER = ["gbs"]
GAME_LABEL = {"gbs": "GBS"}

# dataviz reference palette — fixed categorical slots, one per player count
PLAYER_COLOR = {2: "#2a78d6", 3: "#1baf7a", 4: "#eda100"}
MUTED = "#898781"
GRID = "#e1e0d9"


# ── parsing ──────────────────────────────────────────────────────────────────

def parse_run_id(run_id: str) -> dict | None:
    m = RUN_ID_RE.match(run_id)
    if not m:
        return None
    return {
        "game": m.group("game"),
        "mode": m.group("mode"),
        "players": int(m.group("players")),
        "model": "gpt-oss-20b" if m.group("model") else "Qwen3-4B",
    }


def episode_response_length(jsonl_path: Path) -> float | None:
    """Mean word count of the 'completion' field across all turns in an episode."""
    lengths = []
    if not jsonl_path.exists():
        return None
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            completion = rec.get("completion")
            if completion:
                lengths.append(len(completion.split()))
    return sum(lengths) / len(lengths) if lengths else None


def collect_rows() -> list[dict]:
    rows = []
    for run_dir in sorted(LOGS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_id(run_dir.name)
        if parsed is None:
            continue

        for summary_path in sorted(run_dir.glob("episode_*.summary.json")):
            jsonl_path = summary_path.with_suffix("").with_suffix(".jsonl")
            try:
                with open(summary_path, encoding="utf-8") as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            rows.append({
                "run_id": run_dir.name,
                **parsed,
                "gbs_converged": summary.get("gbs_converged"),
                "gbs_converged_round": summary.get("gbs_converged_round"),
                "mean_response_len_words": episode_response_length(jsonl_path),
            })
    return rows


# ── grouping helpers ─────────────────────────────────────────────────────────

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


# ── plotting ─────────────────────────────────────────────────────────────────

def style_axis(ax, title: str, ylabel: str, modes: list[str]):
    ax.set_title(title, fontsize=10, color="#0b0b0b")
    ax.set_ylabel(ylabel, fontsize=9, color="#52514e")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_LABEL[m] for m in modes], fontsize=9)
    ax.tick_params(axis="y", labelsize=8, colors="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def bar_group(ax, groups: dict, base_key: tuple, modes: list[str], value_fn, label_fmt=None):
    """Draw one cluster of bars (one per player count) at each mode position."""
    n_players = len(PLAYERS_ORDER)
    width = 0.8 / n_players
    for i, n in enumerate(PLAYERS_ORDER):
        xs, ys, errs, labels = [], [], [], []
        for j, mode in enumerate(modes):
            key = base_key + (mode, n)
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
        ax.bar(xs, ys, width=width * 0.9, yerr=errs, capsize=3,
               color=PLAYER_COLOR[n], label=f"{n}p", zorder=3,
               error_kw={"ecolor": "#52514e", "elinewidth": 1})
        if label_fmt:
            for x, y, lab in zip(xs, ys, labels):
                ax.text(x, y, lab, ha="center", va="bottom", fontsize=7, color="#52514e")


def add_player_legend(fig, loc="upper right"):
    handles = [Patch(facecolor=PLAYER_COLOR[n], label=f"{n}p") for n in PLAYERS_ORDER]
    fig.legend(handles=handles, title="players", fontsize=8, title_fontsize=8,
               frameon=False, loc=loc, bbox_to_anchor=(0.99, 0.97) if loc == "upper right" else None)


def add_combined_legend(fig):
    """Player-count patches + reference-line style guide in one figure legend."""
    player_handles = [Patch(facecolor=PLAYER_COLOR[n], label=f"{n}p") for n in PLAYERS_ORDER]
    ref_handles = [
        Line2D([0], [0], color=MUTED, linewidth=1.3, linestyle="--",
               label="noop baseline"),
        Line2D([0], [0], color=MUTED, linewidth=1.3, linestyle=":",
               label="noop+think baseline"),
    ]
    fig.legend(handles=player_handles + ref_handles,
               title="players / baselines", fontsize=8, title_fontsize=8,
               frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.97))


def add_reference_lines(ax, groups: dict, base_key: tuple, modes: list[str], value_fn):
    """Horizontal dashed/dotted lines at noop and noop_thinking means per player.

    Only draws a line if the reference mode is actually in this model's mode list
    (so Qwen3 still gets noop/noop_thinking lines since both are in its grid).
    """
    for mode, ls in REFERENCE_MODES.items():
        if mode not in modes:
            continue
        for n in PLAYERS_ORDER:
            key = base_key + (mode, n)
            grp = groups.get(key, [])
            val, _, _ = value_fn(grp)
            if val is None or val == 0:
                continue
            ax.axhline(val, color=PLAYER_COLOR[n], linewidth=1.1,
                       linestyle=ls, alpha=0.5, zorder=1)


def plot_rounds_to_success(rows: list[dict]):
    gbs_rows = [r for r in rows if r["game"] == "gbs"]
    groups = group_rows(gbs_rows, ("model", "mode", "players"))

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(11, 4.2), sharey=True)
    fig.suptitle("GBS — mean rounds to convergence (successful episodes only)",
                  fontsize=12, color="#0b0b0b")

    def value_fn(grp):
        converged = [r for r in grp if r.get("gbs_converged")]
        val, err = mean_sem([r["gbs_converged_round"] for r in converged])
        label = f"{len(converged)}/{len(grp)}" if grp else ""
        return val, err, label

    for ax, model in zip(axes, MODEL_ORDER):
        modes = MODEL_MODES[model]
        bar_group(ax, groups, (model,), modes, value_fn, label_fmt=True)
        add_reference_lines(ax, groups, (model,), modes, value_fn)
        style_axis(ax, model, "rounds to convergence" if model == MODEL_ORDER[0] else "", modes)

    add_combined_legend(fig)
    fig.text(0.5, 0.01, "Bar labels: converged / total episodes.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "rounds_to_success_gbs.png", dpi=160)
    plt.close(fig)


def plot_response_length(rows: list[dict]):
    gbs_rows = [r for r in rows if r["game"] == "gbs"]
    groups = group_rows(gbs_rows, ("model", "mode", "players"))

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(11, 4.2), sharey=False)
    fig.suptitle("GBS — mean response length (words per completion)",
                 fontsize=12, color="#0b0b0b")

    def value_fn(grp):
        val, err = mean_sem([r["mean_response_len_words"] for r in grp])
        return val, err, ""

    for ax, model in zip(axes, MODEL_ORDER):
        modes = MODEL_MODES[model]
        bar_group(ax, groups, (model,), modes, value_fn)
        add_reference_lines(ax, groups, (model,), modes, value_fn)
        style_axis(ax, model, "words / completion" if model == MODEL_ORDER[0] else "", modes)

    add_combined_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "response_length.png", dpi=160)
    plt.close(fig)


def plot_non_convergence(rows: list[dict]):
    gbs_rows = [r for r in rows if r["game"] == "gbs"]
    groups = group_rows(gbs_rows, ("model", "mode", "players"))

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(11, 4.2), sharey=True)
    fig.suptitle("GBS — fraction of episodes without convergence",
                 fontsize=12, color="#0b0b0b")

    def value_fn(grp):
        if not grp:
            return None, 0.0, ""
        non_conv = sum(1 for r in grp if not r.get("gbs_converged"))
        return non_conv / len(grp), 0.0, f"{non_conv}/{len(grp)}"

    for ax, model in zip(axes, MODEL_ORDER):
        modes = MODEL_MODES[model]
        bar_group(ax, groups, (model,), modes, value_fn, label_fmt=True)
        add_reference_lines(ax, groups, (model,), modes, value_fn)
        style_axis(ax, model,
                   "fraction non-converged" if model == MODEL_ORDER[0] else "", modes)
        ax.set_ylim(0, 1.1)

    add_combined_legend(fig)
    fig.text(0.5, 0.01, "Bar labels: non-converged / total episodes.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "non_convergence.png", dpi=160)
    plt.close(fig)


def plot_n_datapoints(rows: list[dict]):
    gbs_rows = [r for r in rows if r["game"] == "gbs"]
    groups = group_rows(gbs_rows, ("model", "mode", "players"))

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(11, 4.2), sharey=False)
    fig.suptitle("GBS — episodes collected so far (sweep in progress)",
                 fontsize=12, color="#0b0b0b")

    def value_fn(grp):
        return (len(grp), 0.0, str(len(grp))) if grp else (0, 0.0, "0")

    for ax, model in zip(axes, MODEL_ORDER):
        modes = MODEL_MODES[model]
        bar_group(ax, groups, (model,), modes, value_fn, label_fmt=True)
        style_axis(ax, model, "n episodes" if model == MODEL_ORDER[0] else "", modes)

    add_player_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "n_datapoints.png", dpi=160)
    plt.close(fig)


def write_csv(rows: list[dict]):
    fieldnames = ["run_id", "game", "mode", "players", "model",
                  "gbs_converged", "gbs_converged_round", "mean_response_len_words"]
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

    plot_rounds_to_success(rows)
    plot_non_convergence(rows)
    plot_response_length(rows)
    plot_n_datapoints(rows)
    print(f"Wrote figures -> {FIG_DIR}/")


if __name__ == "__main__":
    main()
