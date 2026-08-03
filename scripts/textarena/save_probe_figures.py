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

PROBE_LAYERS = [10, 20, 29]   # all runs probe these layers simultaneously

DEBATE_NOOP_CHUNKS_ID = "debate_noop_chunks_2p"
DEBATE_NOOP_FLAT_ID   = "debate_noop_probe_2p"
MAFIA_NOOP_CHUNKS_ID  = "mafia_noop_chunks_8p"
MAFIA_NOOP_FLAT_ID    = "mafia_noop_probe_8p"
TRAIT_RE = re.compile(r"debate_activation_p1_(.+)_2p|mafia_activation_wolf_(.+)_8p")

# Big-Five / OCEAN traits — pinned to fixed colors so they're identifiable
# across all plots regardless of sort order
OCEAN_COLORS = {
    "openness":      "#2a78d6",  # blue
    "conscientious": "#eb6834",  # orange
    "extraversion":  "#1baf7a",  # green
    "empathy":       "#eda100",  # amber
    "agreeableness": "#e87ba4",  # pink
    "neuroticism":   "#4a3aa7",  # violet
}
OCEAN_TRAITS = set(OCEAN_COLORS)


def parse_target_trait(run_id: str) -> Optional[str]:
    m = TRAIT_RE.match(run_id)
    return (m.group(1) or m.group(2)) if m else None


_VECTORS_DIR = os.path.expandvars(
    "${PERSONA_VECTORS_ROOT}/bf16"
    if os.environ.get("PERSONA_VECTORS_ROOT")
    else "C:/Users/ismyn/UNI/MPI/Thesis/PersVecGen/data/vector_extraction"
       "/persona/qwen3-14b/bf16"
)


def _trait_steer_layer(trait: str) -> Optional[int]:
    """Return the auto-resolved best steering layer for a trait (from companion JSON)."""
    json_path = os.path.join(_VECTORS_DIR, f"{trait}.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path) as f:
            meta = json.load(f)
        sweep = meta.get("sweep_scores", {})
        if not sweep:
            return None

        def _delta(lk: str) -> float:
            scores = {float(k): float(v) for k, v in sweep[lk].items()}
            baseline = scores.get(0.0, min(scores.values()))
            return max(scores.values()) - baseline

        return int(max(sweep.keys(), key=_delta))
    except Exception:
        return None


# ── data loading ──────────────────────────────────────────────────────────────

def _extract_probe(rec: dict, layer: Optional[int] = None) -> Tuple[Dict, List]:
    """Extract (mean_dict, chunks) for the given layer.

    Handles three formats:
      new multi-layer  — {"10": {mean, chunks}, "20": ..., "29": ...}
      old single-layer — {mean: {...}, chunks: [...]}
      oldest flat      — {trait: float, ...}

    When layer is None and the record is multi-layer, the first available
    layer's data is returned (for backward-compat callers that don't care
    about layer).
    """
    probe = rec.get("persona_probe") or {}

    # New multi-layer format: top-level keys are stringified layer ints
    if probe and all(k.isdigit() for k in probe):
        if layer is not None:
            ld = probe.get(str(layer), {})
        else:
            ld = next(iter(probe.values()), {})
        return ld.get("mean", {}), ld.get("chunks", [])

    # Old single-layer format
    if "mean" in probe and isinstance(probe["mean"], dict):
        return probe["mean"], probe.get("chunks", [])

    # Oldest flat format
    flat = {k: v for k, v in probe.items() if isinstance(v, (int, float))}
    return flat, []


def _has_multilayer(rec: dict) -> bool:
    probe = rec.get("persona_probe") or {}
    return bool(probe) and all(k.isdigit() for k in probe)


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

def mean_for_agent(records: List[dict], agent_id: str,
                   layer: Optional[int] = None) -> Dict[str, float]:
    s: Dict[str, float] = defaultdict(float)
    n = 0
    for rec in records:
        if rec["agent_id"] != agent_id:
            continue
        mean_dict, _ = _extract_probe(rec, layer=layer)
        for t, v in mean_dict.items():
            s[t] += v
        n += 1
    return {t: v / n for t, v in s.items()} if n else {}


def mean_wolf_villager(records: List[dict], wolf_map: Dict[int, str],
                       layer: Optional[int] = None,
                       ) -> Tuple[Dict[str, float], Dict[str, float]]:
    ws: Dict[str, float] = defaultdict(float); wn = 0
    vs: Dict[str, float] = defaultdict(float); vn = 0
    for rec in records:
        ep, aid = rec.get("episode", -1), rec["agent_id"]
        mean_dict, _ = _extract_probe(rec, layer=layer)
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
             layer: Optional[int] = None,
             max_tokens: int = 8000, bin_size: int = 10) -> np.ndarray:
    """Cumulative-game chunk trajectory for records matching keep(rec)==True.

    For each episode, concatenates the agent's turns in game order and computes
    per-bin averages using cumulative token positions (turn 1 ends at tok N,
    turn 2 starts at N+1, etc.). Per-episode means are then averaged across
    episodes, so shorter games don't drag down later positions.
    """
    from collections import defaultdict as _dd

    num_buckets = max_tokens // bin_size

    # Group filtered records by (episode, agent_id), preserving file (game) order.
    # Using (ep, agent) as key ensures each player's turns accumulate independently —
    # critical for mafia villagers where 7 players share the same episode.
    by_agent: Dict[Tuple, List[dict]] = _dd(list)
    for rec in records:
        if not keep(rec):
            continue
        key = (rec.get("episode", -1), rec.get("agent_id", ""))
        by_agent[key].append(rec)

    # For each (episode, agent): walk turns in order, accumulate token offsets.
    # Then average across all agents (within same episode first, then across episodes).
    ep_bucket_lists: Dict[int, List[Dict[int, float]]] = _dd(list)

    for (ep, _agent), agent_recs in by_agent.items():
        cum_offset = 0
        agent_buckets: Dict[int, List[float]] = _dd(list)
        for rec in agent_recs:
            _, chunks = _extract_probe(rec, layer=layer)
            for chunk in chunks:
                tok  = chunk.get("token", 0)
                ctok = cum_offset + tok
                cidx = (ctok - 1) // bin_size if ctok > 0 else 0
                if cidx >= num_buckets:
                    continue
                v = chunk.get("scores", {}).get(trait)
                if v is not None:
                    agent_buckets[cidx].append(v)
            if chunks:
                cum_offset += max(c.get("token", 0) for c in chunks)

        # Reduce to per-bucket mean for this agent, then collect by episode.
        ep_bucket_lists[ep].append(
            {cidx: float(np.mean(vals)) for cidx, vals in agent_buckets.items() if vals}
        )

    # Average within each episode (over agents), then across episodes.
    global_buckets: List[List[float]] = [[] for _ in range(num_buckets)]
    for agent_dicts in ep_bucket_lists.values():
        # Episode-mean at each bucket position (average over all agents in this ep).
        all_cidxs = set(cidx for d in agent_dicts for cidx in d)
        for cidx in all_cidxs:
            vals = [d[cidx] for d in agent_dicts if cidx in d]
            if vals and 0 <= cidx < num_buckets:
                global_buckets[cidx].append(float(np.mean(vals)))

    return np.array([np.mean(b) if b else np.nan for b in global_buckets])


def top_traits(means: Dict[str, float], exclude: Optional[str] = None,
               k: Optional[int] = None) -> List[str]:
    ranked = sorted(means, key=lambda t: -abs(means[t]))
    filtered = [t for t in ranked if t != exclude]
    return filtered if k is None else filtered[:k]


# ── plotting helpers ──────────────────────────────────────────────────────────

def _chunk_xs(arr: np.ndarray, bin_size: int = 10) -> np.ndarray:
    """Cumulative token positions corresponding to chunk bin indices."""
    return (np.arange(len(arr)) + 1) * bin_size


def _draw_chunk_ax(ax, records, keep, target_trait, other_traits,
                   noop_mean, label, color, layer: Optional[int] = None):
    """Draw chunk trajectory for all traits + noop reference on an Axes.

    Rendering order (bottom → top):
      thin/faint  — background traits
      semi-bold   — OCEAN traits (always highlighted)
      bold        — target (steered) trait
    """
    # noop reference (horizontal dashed)
    if noop_mean is not None:
        ax.axhline(noop_mean, color=NOOP_COLOR, lw=1.2, ls="--",
                   label=f"noop mean ({noop_mean:.3f})", zorder=1)

    # all other traits — thin, low alpha; OCEAN get pinned colors
    bg_ci = 0  # cycle index for non-OCEAN traits
    for trait in other_traits:
        arr  = chunk_ts(records, keep, trait, layer=layer)
        xs   = _chunk_xs(arr)
        mask = ~np.isnan(arr)
        if not mask.any():
            continue
        if trait in OCEAN_COLORS:
            c = OCEAN_COLORS[trait]
        else:
            c = CAT[bg_ci % len(CAT)]
            bg_ci += 1
        ax.plot(xs[mask], arr[mask], color=c,
                lw=0.6, alpha=0.30, label=trait, zorder=2)

    # target trait — bold on top
    arr  = chunk_ts(records, keep, target_trait, layer=layer)
    xs   = _chunk_xs(arr)
    mask = ~np.isnan(arr)
    if mask.any():
        ax.plot(xs[mask], arr[mask], color=color, lw=TARGET_LW,
                label=f"{target_trait} ★", zorder=4)

    ax.axhline(0, color="#c8c6be", lw=0.7, zorder=0)
    ax.set_title(label, pad=6)
    ax.set_xlabel("Cumulative token position (across all turns)")
    ax.set_ylabel("Projection / ||v||  (>= alpha when steered)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(250))
    ax.legend(loc="upper right", fontsize=6.5, ncol=3,
              handlelength=1.4, columnspacing=1.0, labelspacing=0.35,
              frameon=True, framealpha=0.85)


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


def _noop_debate(base: str) -> Dict[int, Tuple[Dict, Dict, List, bool]]:
    """Return {layer: (p1_means, p0_means, recs, has_chunks)} for all probe layers.

    Loads a single noop run (multi-layer format) and extracts each layer's data.
    """
    recs, has_chunks = _pick_noop(base, DEBATE_NOOP_CHUNKS_ID, DEBATE_NOOP_FLAT_ID)
    return {
        layer: (
            mean_for_agent(recs, "player_1", layer=layer),
            mean_for_agent(recs, "player_0", layer=layer),
            recs,
            has_chunks,
        )
        for layer in PROBE_LAYERS
    }


def _noop_mafia(base: str) -> Dict[int, Tuple[Dict, Dict, Dict, List, bool]]:
    """Return {layer: (wolf_means, vil_means, wolf_map, recs, has_chunks)} for all probe layers."""
    recs, has_chunks = _pick_noop(base, MAFIA_NOOP_CHUNKS_ID, MAFIA_NOOP_FLAT_ID)
    noop_run_id = MAFIA_NOOP_CHUNKS_ID if has_chunks else MAFIA_NOOP_FLAT_ID
    wm = load_wolf_map(os.path.join(base, noop_run_id))
    return {
        layer: (
            *mean_wolf_villager(recs, wm, layer=layer),
            wm,
            recs,
            has_chunks,
        )
        for layer in PROBE_LAYERS
    }


def _draw_noop_chunk_subplots(axes, recs, agent_keep, noop_by_layer):
    """Fill one subplot per PROBE_LAYER with noop chunk trajectories."""
    for row, layer in enumerate(PROBE_LAYERS):
        ax = axes[row]
        p1_means, p0_means, _, _ = noop_by_layer[layer]
        # pick whichever side this axis is for (caller's recs already filtered by agent_keep)
        all_traits = top_traits(p1_means or p0_means)
        bg_ci = 0
        for trait in all_traits:
            arr  = chunk_ts(recs, agent_keep, trait, layer=layer)
            xs   = _chunk_xs(arr)
            mask = ~np.isnan(arr)
            if not mask.any():
                continue
            if trait in OCEAN_COLORS:
                c = OCEAN_COLORS[trait]
            else:
                c = CAT[bg_ci % len(CAT)]
                bg_ci += 1
            ax.plot(xs[mask], arr[mask], color=c, lw=0.6, alpha=0.30,
                    label=trait, zorder=2)
        ax.axhline(0, color="#c8c6be", lw=0.7, zorder=0)
        ax.set_title(f"layer {layer}", pad=6)
        ax.set_xlabel("Cumulative token position (across all turns)")
        ax.set_ylabel("Projection / ||v||")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(250))
        ax.legend(loc="upper right", fontsize=6.5, ncol=3,
                  handlelength=1.4, columnspacing=1.0, labelspacing=0.35,
                  frameon=True, framealpha=0.85)


def debate_figures(logs_dir: str, out_dir: str):
    base = os.path.join(logs_dir, "debate")
    out  = os.path.join(out_dir, "debate")

    noop = _noop_debate(base)

    # noop baseline figures — 3-layer subplots
    for (agent_id, label_prefix, fname) in [
        ("player_1", "player_1 (AGAINST) — noop baseline", "p1_noop.png"),
        ("player_0", "player_0 (FOR) — noop baseline",     "p0_noop.png"),
    ]:
        _, _, noop_recs, has_chunks = noop[PROBE_LAYERS[0]]
        if has_chunks:
            fig, axes = plt.subplots(len(PROBE_LAYERS), 1,
                                     figsize=(14, 6 * len(PROBE_LAYERS)))
            _draw_noop_chunk_subplots(
                axes, noop_recs,
                agent_keep=lambda r, a=agent_id: r["agent_id"] == a,
                noop_by_layer=noop,
            )
            fig.suptitle(f"Debate — {label_prefix}", fontsize=9,
                         color="#9a9890", y=1.002)
        else:
            # No chunks yet — bar chart per layer in a row
            fig, axes = plt.subplots(1, len(PROBE_LAYERS),
                                     figsize=(7 * len(PROBE_LAYERS), 5))
            for col, layer in enumerate(PROBE_LAYERS):
                p1_m, p0_m, _, _ = noop[layer]
                means = p1_m if agent_id == "player_1" else p0_m
                _noop_bar_ax(axes[col], means,
                             f"{label_prefix}\nlayer {layer}\n(chunk data pending)")
            fig.suptitle(f"Debate — {label_prefix}", fontsize=9,
                         color="#9a9890", y=1.002)
        savefig(fig, os.path.join(out, fname))

    # per-experiment chunk figures — additive only, 3 layers each
    for ci, run_dir in enumerate(sorted(glob.glob(
            os.path.join(base, "debate_activation_p1_*_additive_2p")))):
        run_id    = os.path.basename(run_dir)
        trait_raw = parse_target_trait(run_id)
        if not trait_raw:
            continue
        base_trait = trait_raw.removesuffix("_additive")
        recs = load_records(run_dir)
        if not recs:
            continue
        has_chunks = any(len(_extract_probe(r)[1]) > 0 for r in recs[:10])
        if not has_chunks:
            print(f"  skip (no chunks): {run_id}")
            continue

        color = CAT[ci % len(CAT)]
        steer_layer = _trait_steer_layer(base_trait)
        steer_tag   = f"steered @ layer {steer_layer}" if steer_layer else "steered"

        for (agent_id, is_steered, fname) in [
            ("player_1", True,  f"p1_{base_trait}_additive.png"),
            ("player_0", False, f"p0_{base_trait}_additive.png"),
        ]:
            fig, axes = plt.subplots(len(PROBE_LAYERS), 1,
                                     figsize=(14, 6 * len(PROBE_LAYERS)))
            for row, layer in enumerate(PROBE_LAYERS):
                noop_p1, noop_p0, _, _ = noop[layer]
                noop_m = (noop_p1 if agent_id == "player_1" else noop_p0).get(base_trait)
                means  = mean_for_agent(recs, agent_id, layer=layer)
                others = top_traits(means, exclude=base_trait)
                role_tag = steer_tag if is_steered else "unsteered"
                _draw_chunk_ax(
                    axes[row], recs,
                    keep=lambda r, a=agent_id: r["agent_id"] == a,
                    target_trait=base_trait, other_traits=others,
                    noop_mean=noop_m,
                    label=f"{agent_id} [{role_tag}] — {base_trait} | probe layer {layer}",
                    color=color, layer=layer,
                )
            fig.suptitle(f"Debate — {run_id}  [{steer_tag}]", fontsize=9,
                         color="#9a9890", y=1.002)
            savefig(fig, os.path.join(out, fname))


# ── mafia figures ─────────────────────────────────────────────────────────────

def mafia_figures(logs_dir: str, out_dir: str):
    base = os.path.join(logs_dir, "mafia")
    out  = os.path.join(out_dir, "mafia")

    noop = _noop_mafia(base)

    # noop baseline figures — 3-layer subplots
    _, _, noop_wm, noop_recs, noop_has_chunks = noop[PROBE_LAYERS[0]]
    for (is_wolf, label_prefix, fname) in [
        (True,  "Wolf — noop baseline",     "wolf_noop.png"),
        (False, "Villager — noop baseline", "vil_noop.png"),
    ]:
        agent_keep = lambda r, wm=noop_wm, iw=is_wolf: (
            r["agent_id"] == wm.get(r.get("episode", -1))
        ) == iw

        if noop_has_chunks:
            fig, axes = plt.subplots(len(PROBE_LAYERS), 1,
                                     figsize=(14, 6 * len(PROBE_LAYERS)))
            for row, layer in enumerate(PROBE_LAYERS):
                ax = axes[row]
                wolf_m, vil_m, _, _, _ = noop[layer]
                means     = wolf_m if is_wolf else vil_m
                all_traits = top_traits(means)
                bg_ci = 0
                for trait in all_traits:
                    arr  = chunk_ts(noop_recs, agent_keep, trait, layer=layer)
                    xs   = _chunk_xs(arr)
                    mask = ~np.isnan(arr)
                    if not mask.any():
                        continue
                    if trait in OCEAN_COLORS:
                        c = OCEAN_COLORS[trait]
                    else:
                        c = CAT[bg_ci % len(CAT)]
                        bg_ci += 1
                    ax.plot(xs[mask], arr[mask], color=c, lw=0.6, alpha=0.30,
                            label=trait, zorder=2)
                ax.axhline(0, color="#c8c6be", lw=0.7, zorder=0)
                ax.set_title(f"{label_prefix} | layer {layer}", pad=6)
                ax.set_xlabel("Cumulative token position (across all turns)")
                ax.set_ylabel("Projection / ||v||")
                ax.xaxis.set_major_locator(ticker.MultipleLocator(250))
                ax.legend(loc="upper right", fontsize=6.5, ncol=3,
                          handlelength=1.4, columnspacing=1.0, labelspacing=0.35,
                          frameon=True, framealpha=0.85)
        else:
            fig, axes = plt.subplots(1, len(PROBE_LAYERS),
                                     figsize=(7 * len(PROBE_LAYERS), 5))
            for col, layer in enumerate(PROBE_LAYERS):
                wolf_m, vil_m, _, _, _ = noop[layer]
                means = wolf_m if is_wolf else vil_m
                _noop_bar_ax(axes[col], means,
                             f"{label_prefix}\nlayer {layer}\n(chunk data pending)")
        fig.suptitle(f"Mafia — {label_prefix}", fontsize=9,
                     color="#9a9890", y=1.002)
        savefig(fig, os.path.join(out, fname))

    # per-experiment chunk figures — additive only, 3 layers each
    for ci, run_dir in enumerate(sorted(glob.glob(
            os.path.join(base, "mafia_activation_wolf_*_additive_8p")))):
        run_id    = os.path.basename(run_dir)
        trait_raw = parse_target_trait(run_id)
        if not trait_raw:
            continue
        base_trait = trait_raw.removesuffix("_additive")
        recs = load_records(run_dir)
        if not recs:
            continue
        has_chunks = any(len(_extract_probe(r)[1]) > 0 for r in recs[:10])
        if not has_chunks:
            print(f"  skip (no chunks): {run_id}")
            continue

        wolf_map    = load_wolf_map(run_dir)
        color       = CAT[ci % len(CAT)]
        steer_layer = _trait_steer_layer(base_trait)
        steer_tag   = f"steered @ layer {steer_layer}" if steer_layer else "steered"

        for (is_wolf, role_label, fname) in [
            (True,  f"wolf [{steer_tag}]",    f"wolf_{base_trait}_additive.png"),
            (False, "villager [unsteered]",    f"vil_{base_trait}_additive.png"),
        ]:
            fig, axes = plt.subplots(len(PROBE_LAYERS), 1,
                                     figsize=(14, 6 * len(PROBE_LAYERS)))
            for row, layer in enumerate(PROBE_LAYERS):
                noop_wolf_m, noop_vil_m, _, _, _ = noop[layer]
                noop_m = (noop_wolf_m if is_wolf else noop_vil_m).get(base_trait)
                means  = mean_wolf_villager(recs, wolf_map, layer=layer)[0 if is_wolf else 1]
                others = top_traits(means, exclude=base_trait)
                _draw_chunk_ax(
                    axes[row], recs,
                    keep=lambda r, wm=wolf_map, iw=is_wolf:
                        (r["agent_id"] == wm.get(r.get("episode", -1))) == iw,
                    target_trait=base_trait, other_traits=others,
                    noop_mean=noop_m,
                    label=f"{role_label} — {base_trait} | probe layer {layer}",
                    color=color, layer=layer,
                )
            fig.suptitle(f"Mafia — {run_id}  [{steer_tag}]", fontsize=9,
                         color="#9a9890", y=1.002)
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
