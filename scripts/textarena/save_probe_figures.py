#!/usr/bin/env python3
"""Save persona probe figures as PNG files.

For each game × steering condition, saves two figures:
  <game>/p1_<trait>_steered.png  or  <game>/wolf_<trait>_steered.png
  <game>/p0_<trait>_steered.png  or  <game>/vil_<trait>_steered.png

Plus noop baseline figures (bar charts, no chunk data available):
  <game>/p1_noop.png / <game>/p0_noop.png
  <game>/wolf_noop.png / <game>/vil_noop.png

Each chunk figure shows:
  - Chunk trajectory of target trait (bold) + top-4 other traits
  - Noop mean for the target trait as a dashed horizontal reference
  - X axis: token position; Y axis: cosine projection

Usage
-----
python scripts/textarena/save_probe_figures.py --logs-dir logs --out-dir reports/figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.titleweight":      "semibold",
    "axes.labelsize":        9,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.alpha":            0.25,
    "grid.linewidth":        0.6,
    "legend.fontsize":       8,
    "legend.frameon":        False,
    "figure.dpi":            150,
    "savefig.dpi":           150,
    "savefig.bbox":          "tight",
    "savefig.facecolor":     "#f5f4f0",
    "figure.facecolor":      "#f5f4f0",
    "axes.facecolor":        "#ffffff",
})

# Validated categorical palette (adjacent-pairs safe)
CAT = ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7","#e34948"]
NOOP_COLOR  = "#9a9890"
TARGET_LW   = 2.2
OTHER_LW    = 1.1
OTHER_ALPHA = 0.55

# ── constants ─────────────────────────────────────────────────────────────────

# Prefer the new-format noop runs (with chunk time-series).
# Falls back to the old flat-dict runs if the chunk version hasn't been run yet.
DEBATE_NOOP_CHUNKS_ID = "debate_noop_chunks_2p"
DEBATE_NOOP_FLAT_ID   = "debate_noop_probe_2p"
MAFIA_NOOP_CHUNKS_ID  = "mafia_noop_chunks_8p"
MAFIA_NOOP_FLAT_ID    = "mafia_noop_probe_8p"
TRAIT_RE = re.compile(r"debate_activation_p1_(.+)_2p|mafia_activation_wolf_(.+)_8p")


def parse_target_trait(run_id: str) -> Optional[str]:
    m = TRAIT_RE.match(run_id)
    return (m.group(1) or m.group(2)) if m else None


# ── data loading ──────────────────────────────────────────────────────────────

def _extract_probe(rec: dict) -> Tuple[Dict, List]:
    probe = rec.get("persona_probe") or {}
    if "mean" in probe and isinstance(probe["mean"], dict):
        return probe["mean"], probe.get("chunks", [])
    flat = {k: v for k, v in probe.items() if isinstance(v, (int, float))}
    return flat, []


def load_records(run_dir: str) -> List[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(run_dir, "episode_*.jsonl"))):
        try:
            for line in open(path, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    mean, _ = _extract_probe(rec)
                    if mean:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue
    return records


def load_wolf_map(run_dir: str) -> Dict[int, str]:
    wolf_map: Dict[int, str] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "episode_*.summary.json"))):
        m = re.search(r"episode_(\d+)\.summary\.json$", path)
        if not m:
            continue
        ep = int(m.group(1))
        try:
            s = json.load(open(path, encoding="utf-8"))
            for idx, info in s.get("final_info", {}).get("close_info", {}).items():
                if info.get("role") == "Mafia":
                    wolf_map[ep] = f"player_{idx}"
                    break
        except (OSError, json.JSONDecodeError):
            continue
    return wolf_map


# ── aggregation ───────────────────────────────────────────────────────────────

def mean_for_agent(records: List[dict], agent_id: str) -> Dict[str, float]:
    s: Dict[str, float] = defaultdict(float)
    n = 0
    for rec in records:
        if rec["agent_id"] != agent_id:
            continue
        mean_dict, _ = _extract_probe(rec)
        for t, v in mean_dict.items():
            s[t] += v
        n += 1
    return {t: v / n for t, v in s.items()} if n else {}


def mean_wolf_villager(records: List[dict], wolf_map: Dict[int, str]
                       ) -> Tuple[Dict[str, float], Dict[str, float]]:
    ws: Dict[str, float] = defaultdict(float); wn = 0
    vs: Dict[str, float] = defaultdict(float); vn = 0
    for rec in records:
        ep, aid = rec.get("episode", -1), rec["agent_id"]
        mean_dict, _ = _extract_probe(rec)
        if aid == wolf_map.get(ep):
            for t, v in mean_dict.items(): ws[t] += v
            wn += 1
        else:
            for t, v in mean_dict.items(): vs[t] += v
            vn += 1
    wolf_m = {t: v / wn for t, v in ws.items()} if wn else {}
    vil_m  = {t: v / vn for t, v in vs.items()} if vn else {}
    return wolf_m, vil_m


def chunk_ts(records: List[dict], keep: callable, trait: str,
             max_chunks: int = 30) -> np.ndarray:
    """Chunk trajectory for records matching keep(rec)==True, one trait."""
    buckets: List[List[float]] = [[] for _ in range(max_chunks)]
    for rec in records:
        if not keep(rec):
            continue
        _, chunks = _extract_probe(rec)
        for chunk in chunks:
            token = chunk.get("token", 0)
            cidx  = max(0, (token // 10) - 1)
            if cidx < max_chunks:
                v = chunk.get("scores", {}).get(trait)
                if v is not None:
                    buckets[cidx].append(v)
    return np.array([
        np.mean(b) if b else np.nan for b in buckets
    ])


def top_traits(means: Dict[str, float], exclude: Optional[str] = None,
               k: int = 4) -> List[str]:
    ranked = sorted(means, key=lambda t: -abs(means[t]))
    return [t for t in ranked if t != exclude][:k]


# ── plotting helpers ──────────────────────────────────────────────────────────

def _chunk_xs(arr: np.ndarray) -> np.ndarray:
    """Token positions corresponding to chunk indices."""
    return (np.arange(len(arr)) + 1) * 10


def _draw_chunk_ax(ax, records, keep, target_trait, other_traits,
                   noop_mean, label, color):
    """Draw chunk trajectory + noop reference on an Axes."""
    xs_full = _chunk_xs(np.zeros(30))

    # noop reference (horizontal dashed)
    if noop_mean is not None:
        ax.axhline(noop_mean, color=NOOP_COLOR, lw=1.2, ls="--",
                   label=f"noop mean ({noop_mean:.3f})", zorder=1)

    # other traits (thin, muted)
    for ti, trait in enumerate(other_traits):
        arr = chunk_ts(records, keep, trait)
        xs  = _chunk_xs(arr)
        mask = ~np.isnan(arr)
        if mask.any():
            ax.plot(xs[mask], arr[mask], color=CAT[(ti + 1) % len(CAT)],
                    lw=OTHER_LW, alpha=OTHER_ALPHA, label=trait, zorder=2)

    # target trait (bold)
    arr  = chunk_ts(records, keep, target_trait)
    xs   = _chunk_xs(arr)
    mask = ~np.isnan(arr)
    if mask.any():
        ax.plot(xs[mask], arr[mask], color=color, lw=TARGET_LW,
                label=f"{target_trait} ★", zorder=3)

    ax.axhline(0, color="#c8c6be", lw=0.7, zorder=0)
    ax.set_title(label, pad=6)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Projection / ||v||  (>= alpha when steered)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.legend(loc="upper right", fontsize=7.5)


def _noop_bar_ax(ax, means, title, top_k=12):
    """Bar chart of top-K mean trait projections for noop baseline."""
    ranked = sorted(means, key=lambda t: -abs(means[t]))[:top_k]
    vals   = [means[t] for t in ranked]
    colors = [CAT[0] if v >= 0 else "#e34948" for v in vals]
    ys     = range(len(ranked))
    ax.barh(ys, vals, color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(ranked, fontsize=8)
    ax.axvline(0, color="#c8c6be", lw=0.8)
    ax.set_xlabel("Mean projection / ||v||")
    ax.set_title(title, pad=6)
    ax.invert_yaxis()


def savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")


# ── debate figures ────────────────────────────────────────────────────────────

def _pick_noop(base: str, chunks_id: str, flat_id: str) -> Tuple[List[dict], bool]:
    """Load noop records, preferring the chunk-format run if it exists."""
    chunks_dir = os.path.join(base, chunks_id)
    recs = load_records(chunks_dir)
    if recs and any(len(_extract_probe(r)[1]) > 0 for r in recs[:10]):
        print(f"  noop: using chunk-format run ({chunks_id})")
        return recs, True
    flat_recs = load_records(os.path.join(base, flat_id))
    if flat_recs:
        print(f"  noop: chunk run not ready — falling back to flat-format ({flat_id})")
    return flat_recs, False


def debate_figures(logs_dir: str, out_dir: str):
    base = os.path.join(logs_dir, "debate")
    out  = os.path.join(out_dir, "debate")

    # noop — prefer new chunk-format run
    noop_recs, noop_has_chunks = _pick_noop(
        base, DEBATE_NOOP_CHUNKS_ID, DEBATE_NOOP_FLAT_ID
    )
    noop_p1 = mean_for_agent(noop_recs, "player_1")
    noop_p0 = mean_for_agent(noop_recs, "player_0")

    # noop baseline figures
    for (means, label, fname, agent_id) in [
        (noop_p1, "player_1 (AGAINST) — noop baseline", "p1_noop.png", "player_1"),
        (noop_p0, "player_0 (FOR) — noop baseline",     "p0_noop.png", "player_0"),
    ]:
        if noop_has_chunks:
            # chunk trajectory of top-5 traits
            top5 = top_traits(means, k=5)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for ti, trait in enumerate(top5):
                arr  = chunk_ts(noop_recs, lambda r, a=agent_id: r["agent_id"] == a, trait)
                xs   = _chunk_xs(arr)
                mask = ~np.isnan(arr)
                if mask.any():
                    ax.plot(xs[mask], arr[mask], color=CAT[ti % len(CAT)],
                            lw=TARGET_LW if ti == 0 else OTHER_LW,
                            alpha=1.0 if ti == 0 else OTHER_ALPHA,
                            label=trait)
            ax.axhline(0, color="#c8c6be", lw=0.7, zorder=0)
            ax.set_title(label, pad=6)
            ax.set_xlabel("Token position")
            ax.set_ylabel("Projection / ||v||  (>= alpha when steered)")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.legend(loc="upper right", fontsize=7.5)
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            _noop_bar_ax(ax, means, label + "\n(chunk data pending — re-run after overnight job)")
        fig.suptitle("Debate — noop baseline", fontsize=9, color="#9a9890", y=1.0)
        savefig(fig, os.path.join(out, fname))

    # per-experiment chunk figures
    for ci, run_dir in enumerate(sorted(glob.glob(
            os.path.join(base, "debate_activation_p1_*")))):
        run_id = os.path.basename(run_dir)
        trait  = parse_target_trait(run_id)
        if not trait:
            continue
        recs = load_records(run_dir)
        if not recs:
            continue
        has_chunks = any(len(_extract_probe(r)[1]) > 0 for r in recs[:10])
        if not has_chunks:
            print(f"  skip (no chunks): {run_id}")
            continue

        color   = CAT[ci % len(CAT)]
        p1_mean = mean_for_agent(recs, "player_1")
        p0_mean = mean_for_agent(recs, "player_0")

        for (agent_id, means, noop_m, label, fname) in [
            ("player_1", p1_mean, noop_p1.get(trait),
             f"player_1 — {trait} (steered)",  f"p1_{trait}.png"),
            ("player_0", p0_mean, noop_p0.get(trait),
             f"player_0 — {trait} run (unsteered)", f"p0_{trait}.png"),
        ]:
            others = top_traits(means, exclude=trait, k=4)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _draw_chunk_ax(
                ax, recs,
                keep=lambda r, a=agent_id: r["agent_id"] == a,
                target_trait=trait, other_traits=others,
                noop_mean=noop_m, label=label, color=color,
            )
            fig.suptitle(f"Debate — {run_id}", fontsize=9,
                         color="#9a9890", y=1.0)
            savefig(fig, os.path.join(out, fname))


# ── mafia figures ─────────────────────────────────────────────────────────────

def mafia_figures(logs_dir: str, out_dir: str):
    base = os.path.join(logs_dir, "mafia")
    out  = os.path.join(out_dir, "mafia")

    # noop — prefer new chunk-format run
    noop_chunks_dir = os.path.join(base, MAFIA_NOOP_CHUNKS_ID)
    noop_recs, noop_has_chunks = _pick_noop(
        base, MAFIA_NOOP_CHUNKS_ID, MAFIA_NOOP_FLAT_ID
    )
    noop_wolf_map = load_wolf_map(
        noop_chunks_dir if noop_has_chunks else os.path.join(base, MAFIA_NOOP_FLAT_ID)
    )
    noop_wolf_m, noop_vil_m = mean_wolf_villager(noop_recs, noop_wolf_map)

    # noop baseline figures
    for (means, is_wolf, label, fname) in [
        (noop_wolf_m, True,  "Wolf — noop baseline",     "wolf_noop.png"),
        (noop_vil_m,  False, "Villager — noop baseline", "vil_noop.png"),
    ]:
        if noop_has_chunks:
            top5 = top_traits(means, k=5)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for ti, trait in enumerate(top5):
                arr  = chunk_ts(
                    noop_recs,
                    keep=lambda r, wm=noop_wolf_map, iw=is_wolf:
                        (r["agent_id"] == wm.get(r.get("episode", -1))) == iw,
                    trait=trait,
                )
                xs   = _chunk_xs(arr)
                mask = ~np.isnan(arr)
                if mask.any():
                    ax.plot(xs[mask], arr[mask], color=CAT[ti % len(CAT)],
                            lw=TARGET_LW if ti == 0 else OTHER_LW,
                            alpha=1.0 if ti == 0 else OTHER_ALPHA,
                            label=trait)
            ax.axhline(0, color="#c8c6be", lw=0.7, zorder=0)
            ax.set_title(label, pad=6)
            ax.set_xlabel("Token position")
            ax.set_ylabel("Projection / ||v||  (>= alpha when steered)")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.legend(loc="upper right", fontsize=7.5)
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            _noop_bar_ax(ax, means, label + "\n(chunk data pending — re-run after overnight job)")
        fig.suptitle("Mafia — noop baseline", fontsize=9, color="#9a9890", y=1.0)
        savefig(fig, os.path.join(out, fname))

    # per-experiment chunk figures
    for ci, run_dir in enumerate(sorted(glob.glob(
            os.path.join(base, "mafia_activation_wolf_*")))):
        run_id = os.path.basename(run_dir)
        trait  = parse_target_trait(run_id)
        if not trait:
            continue
        recs = load_records(run_dir)
        if not recs:
            continue
        has_chunks = any(len(_extract_probe(r)[1]) > 0 for r in recs[:10])
        if not has_chunks:
            print(f"  skip (no chunks): {run_id}")
            continue

        wolf_map = load_wolf_map(run_dir)
        wolf_m, vil_m = mean_wolf_villager(recs, wolf_map)
        color = CAT[ci % len(CAT)]

        for (is_wolf, means, noop_m, label, fname) in [
            (True,  wolf_m, noop_wolf_m.get(trait),
             f"Wolf — {trait} (steered)",        f"wolf_{trait}.png"),
            (False, vil_m,  noop_vil_m.get(trait),
             f"Villager — {trait} run (unsteered)", f"vil_{trait}.png"),
        ]:
            others = top_traits(means, exclude=trait, k=4)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _draw_chunk_ax(
                ax, recs,
                keep=lambda r, wm=wolf_map, iw=is_wolf:
                    (r["agent_id"] == wm.get(r.get("episode", -1))) == iw,
                target_trait=trait, other_traits=others,
                noop_mean=noop_m, label=label, color=color,
            )
            fig.suptitle(f"Mafia — {run_id}", fontsize=9,
                         color="#9a9890", y=1.0)
            savefig(fig, os.path.join(out, fname))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="Save probe chunk figures as PNG files.")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--out-dir",  default="reports/figures")
    args = ap.parse_args(argv)

    print("=== Debate ===")
    debate_figures(args.logs_dir, args.out_dir)

    print("=== Mafia ===")
    mafia_figures(args.logs_dir, args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
