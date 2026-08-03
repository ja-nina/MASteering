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

# Noop run IDs keyed by probe layer.  Each steered run's probe layer is looked
# up from the companion JSON metadata so the reference line is always from the
# correct layer.  Falls back to the flat-format run if no chunk run exists yet.
DEBATE_NOOP_BY_LAYER = {
    10: "debate_noop_chunks_layer10_2p",
    20: "debate_noop_chunks_layer20_2p",
    29: "debate_noop_chunks_2p",
}
MAFIA_NOOP_BY_LAYER = {
    10: "mafia_noop_chunks_layer10_8p",
    20: "mafia_noop_chunks_layer20_8p",
    29: "mafia_noop_chunks_8p",
}
DEBATE_NOOP_FLAT_ID = "debate_noop_probe_2p"
MAFIA_NOOP_FLAT_ID  = "mafia_noop_probe_8p"
TRAIT_RE = re.compile(r"debate_activation_p1_(.+)_2p|mafia_activation_wolf_(.+)_8p")

# Per-trait best probe layer, loaded once from the companion JSON files.
# Populated by _load_trait_layer_map() on first use.
_TRAIT_LAYER_MAP: Dict[str, int] = {}


def _load_trait_layer_map(vectors_dir: str) -> Dict[str, int]:
    """Return {trait: best_layer_int} from PersVecGen companion JSON files."""
    import pathlib
    result: Dict[str, int] = {}
    for jf in sorted(pathlib.Path(vectors_dir).glob("*.json")):
        trait = jf.stem
        try:
            meta = json.load(open(str(jf)))
            sweep = meta.get("sweep_scores", {})
            if not sweep:
                continue
            def _delta(lk: str, sw=sweep) -> float:
                sv = {float(k): float(v) for k, v in sw[lk].items()}
                b = sv.get(0.0, min(sv.values()))
                return max(sv.values()) - b
            result[trait] = int(max(sweep.keys(), key=_delta))
        except Exception:
            continue
    return result


def trait_probe_layer(trait: str, vectors_dir: str) -> int:
    """Return the optimal probe layer for a trait (cached after first call)."""
    global _TRAIT_LAYER_MAP
    if not _TRAIT_LAYER_MAP:
        _TRAIT_LAYER_MAP = _load_trait_layer_map(vectors_dir)
    return _TRAIT_LAYER_MAP.get(trait, 29)  # default to 29 if unknown

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
            _, chunks = _extract_probe(rec)
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
                   noop_mean, label, color):
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
        arr  = chunk_ts(records, keep, trait)
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
    arr  = chunk_ts(records, keep, target_trait)
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


def _noop_cache_debate(base: str) -> Dict[int, Tuple[Dict, Dict, bool]]:
    """Return {layer: (noop_p1_means, noop_p0_means, has_chunks)} for all layers."""
    cache: Dict[int, Tuple[Dict, Dict, bool]] = {}
    for layer, chunks_id in DEBATE_NOOP_BY_LAYER.items():
        recs, has_chunks = _pick_noop(base, chunks_id, DEBATE_NOOP_FLAT_ID)
        cache[layer] = (
            mean_for_agent(recs, "player_1"),
            mean_for_agent(recs, "player_0"),
            has_chunks,
        )
    return cache


def _noop_cache_mafia(base: str) -> Dict[int, Tuple[Dict, Dict, Dict, bool]]:
    """Return {layer: (wolf_means, vil_means, wolf_map, has_chunks)} for all layers."""
    cache: Dict[int, Tuple[Dict, Dict, Dict, bool]] = {}
    for layer, chunks_id in MAFIA_NOOP_BY_LAYER.items():
        recs, has_chunks = _pick_noop(base, chunks_id, MAFIA_NOOP_FLAT_ID)
        wm = load_wolf_map(
            os.path.join(base, chunks_id if has_chunks else MAFIA_NOOP_FLAT_ID)
        )
        wolf_m, vil_m = mean_wolf_villager(recs, wm)
        cache[layer] = (wolf_m, vil_m, wm, recs, has_chunks)
    return cache


def debate_figures(logs_dir: str, out_dir: str):
    base = os.path.join(logs_dir, "debate")
    out  = os.path.join(out_dir, "debate")

    vectors_dir = os.path.expandvars(
        "${PERSONA_VECTORS_ROOT}/bf16" if os.environ.get("PERSONA_VECTORS_ROOT")
        else "C:/Users/ismyn/UNI/MPI/Thesis/PersVecGen/data/vector_extraction/persona/qwen3-14b/bf16"
    )

    # Load noop runs for all three probe layers
    noop = _noop_cache_debate(base)
    # Use layer-29 noop for the generic noop baseline figures
    noop_p1_29, noop_p0_29, noop_has_chunks_29 = noop[29]

    # noop baseline figures (layer 29 only — the generic reference)
    for (means, label, fname, agent_id) in [
        (noop_p1_29, "player_1 (AGAINST) — noop baseline", "p1_noop.png", "player_1"),
        (noop_p0_29, "player_0 (FOR) — noop baseline",     "p0_noop.png", "player_0"),
    ]:
        if noop_has_chunks_29:
            all_traits = top_traits(means)
            fig, ax = plt.subplots(figsize=(14, 7))
            recs_29, _ = _pick_noop(base, DEBATE_NOOP_BY_LAYER[29], DEBATE_NOOP_FLAT_ID)
            bg_ci = 0
            for trait in all_traits:
                arr  = chunk_ts(recs_29, lambda r, a=agent_id: r["agent_id"] == a, trait)
                xs   = _chunk_xs(arr)
                mask = ~np.isnan(arr)
                if not mask.any():
                    continue
                c = OCEAN_COLORS[trait] if trait in OCEAN_COLORS else CAT[bg_ci % len(CAT)]
                if trait not in OCEAN_COLORS:
                    bg_ci += 1
                ax.plot(xs[mask], arr[mask], color=c,
                        lw=0.6, alpha=0.30, label=trait, zorder=2)
            ax.axhline(0, color="#c8c6be", lw=0.7, zorder=0)
            ax.set_title(label, pad=6)
            ax.set_xlabel("Cumulative token position (across all turns)")
            ax.set_ylabel("Projection / ||v||  (>= alpha when steered)")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(250))
            ax.legend(loc="upper right", fontsize=6.5, ncol=3,
                      handlelength=1.4, columnspacing=1.0, labelspacing=0.35,
                      frameon=True, framealpha=0.85)
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            _noop_bar_ax(ax, means, label + "\n(chunk data pending — re-run after overnight job)")
        fig.suptitle("Debate — noop baseline", fontsize=9, color="#9a9890", y=1.0)
        savefig(fig, os.path.join(out, fname))

    # per-experiment chunk figures — additive only
    for ci, run_dir in enumerate(sorted(glob.glob(
            os.path.join(base, "debate_activation_p1_*_additive_2p")))):
        run_id    = os.path.basename(run_dir)
        trait_raw = parse_target_trait(run_id)   # e.g. "charismatic_additive"
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

        # Pick noop means from the layer matching this trait's probe layer
        probe_layer = trait_probe_layer(base_trait, vectors_dir)
        noop_p1, noop_p0, _ = noop.get(probe_layer, noop[29])

        color   = CAT[ci % len(CAT)]
        p1_mean = mean_for_agent(recs, "player_1")
        p0_mean = mean_for_agent(recs, "player_0")

        for (agent_id, means, noop_m, label, fname) in [
            ("player_1", p1_mean, noop_p1.get(base_trait),
             f"player_1 — {base_trait} [additive] (steered, probe layer {probe_layer})",
             f"p1_{base_trait}_additive.png"),
            ("player_0", p0_mean, noop_p0.get(base_trait),
             f"player_0 — {base_trait} [additive] run (unsteered)",
             f"p0_{base_trait}_additive.png"),
        ]:
            others = top_traits(means, exclude=base_trait)
            fig, ax = plt.subplots(figsize=(14, 7))
            _draw_chunk_ax(
                ax, recs,
                keep=lambda r, a=agent_id: r["agent_id"] == a,
                target_trait=base_trait, other_traits=others,
                noop_mean=noop_m, label=label, color=color,
            )
            fig.suptitle(f"Debate — {run_id}", fontsize=9,
                         color="#9a9890", y=1.0)
            savefig(fig, os.path.join(out, fname))


# ── mafia figures ─────────────────────────────────────────────────────────────

def mafia_figures(logs_dir: str, out_dir: str):
    base = os.path.join(logs_dir, "mafia")
    out  = os.path.join(out_dir, "mafia")

    vectors_dir = os.path.expandvars(
        "${PERSONA_VECTORS_ROOT}/bf16" if os.environ.get("PERSONA_VECTORS_ROOT")
        else "C:/Users/ismyn/UNI/MPI/Thesis/PersVecGen/data/vector_extraction/persona/qwen3-14b/bf16"
    )

    # Load noop runs for all three probe layers
    noop = _noop_cache_mafia(base)
    noop_wolf_29, noop_vil_29, noop_wm_29, noop_recs_29, noop_has_chunks_29 = noop[29]

    # noop baseline figures (layer 29 only — the generic reference)
    for (means, is_wolf, label, fname) in [
        (noop_wolf_29, True,  "Wolf — noop baseline",     "wolf_noop.png"),
        (noop_vil_29,  False, "Villager — noop baseline", "vil_noop.png"),
    ]:
        if noop_has_chunks_29:
            all_traits = top_traits(means)
            fig, ax = plt.subplots(figsize=(14, 7))
            bg_ci = 0
            for trait in all_traits:
                arr  = chunk_ts(
                    noop_recs_29,
                    keep=lambda r, wm=noop_wm_29, iw=is_wolf:
                        (r["agent_id"] == wm.get(r.get("episode", -1))) == iw,
                    trait=trait,
                )
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
            ax.axhline(0, color="#c8c6be", lw=0.7, zorder=0)
            ax.set_title(label, pad=6)
            ax.set_xlabel("Cumulative token position (across all turns)")
            ax.set_ylabel("Projection / ||v||  (>= alpha when steered)")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(250))
            ax.legend(loc="upper right", fontsize=6.5, ncol=3,
                      handlelength=1.4, columnspacing=1.0, labelspacing=0.35,
                      frameon=True, framealpha=0.85)
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            _noop_bar_ax(ax, means, label + "\n(chunk data pending — re-run after overnight job)")
        fig.suptitle("Mafia — noop baseline", fontsize=9, color="#9a9890", y=1.0)
        savefig(fig, os.path.join(out, fname))

    # per-experiment chunk figures — additive only
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

        # Pick noop means from the layer matching this trait's probe layer
        probe_layer = trait_probe_layer(base_trait, vectors_dir)
        noop_wolf_m, noop_vil_m, _, _, _ = noop.get(probe_layer, noop[29])

        wolf_map = load_wolf_map(run_dir)
        wolf_m, vil_m = mean_wolf_villager(recs, wolf_map)
        color = CAT[ci % len(CAT)]

        for (is_wolf, means, noop_m, label, fname) in [
            (True,  wolf_m, noop_wolf_m.get(base_trait),
             f"Wolf — {base_trait} [additive] (steered, probe layer {probe_layer})",
             f"wolf_{base_trait}_additive.png"),
            (False, vil_m,  noop_vil_m.get(base_trait),
             f"Villager — {base_trait} [additive] run (unsteered)",
             f"vil_{base_trait}_additive.png"),
        ]:
            others = top_traits(means, exclude=base_trait)
            fig, ax = plt.subplots(figsize=(14, 7))
            _draw_chunk_ax(
                ax, recs,
                keep=lambda r, wm=wolf_map, iw=is_wolf:
                    (r["agent_id"] == wm.get(r.get("episode", -1))) == iw,
                target_trait=base_trait, other_traits=others,
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
