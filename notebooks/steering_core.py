from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # must be before any pyplot import; needed on headless servers

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScheduleEntry:
    trait: str
    layer: int
    start: int
    end: Optional[int]   # None = steer until end of generation
    coeff: float
    mode: str = "additive"  # "additive" | "adaptive"


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------

def load_vectors(vectors_dir: str) -> Dict[str, Dict[int, torch.Tensor]]:
    """Load all *.pt trait vectors from a PersVecGen bf16 directory.

    Returns {trait_name: {layer_int: tensor}}.
    Plain-tensor files (not dicts) are stored as {-1: tensor}.
    """
    import logging
    _log = logging.getLogger("steering_core")
    p = pathlib.Path(vectors_dir)
    _log.info("load_vectors: scanning %s (exists=%s)", p, p.exists())
    if p.exists():
        all_files = list(p.iterdir())
        _log.info("load_vectors: directory contains %d entries: %s",
                  len(all_files), [f.name for f in all_files[:20]])
    pt_files = sorted(p.glob("*.pt"))
    _log.info("load_vectors: found %d .pt files", len(pt_files))
    result: Dict[str, Dict[int, torch.Tensor]] = {}
    for pt_path in pt_files:
        trait = pt_path.stem
        loaded = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            layer_keys = list(loaded.keys())
            result[trait] = {int(k): v.float() for k, v in loaded.items()}
            _log.info("  loaded trait=%r layers=%s", trait, [int(k) for k in layer_keys])
        else:
            result[trait] = {-1: loaded.float()}
            _log.info("  loaded trait=%r as plain tensor", trait)
    _log.info("load_vectors: total traits loaded = %d", len(result))
    return result


def best_layer(trait: str, vectors_dir: str) -> Optional[int]:
    """Return the integer layer with the highest sweep delta for a trait.

    Reads the companion <trait>.json file (PersVecGen metadata).
    Returns None if the file does not exist.
    """
    json_path = pathlib.Path(vectors_dir) / f"{trait}.json"
    if not json_path.exists():
        return None
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_id: str = "Qwen/Qwen3-14B",
    bits: Optional[int] = 4,
) -> Tuple[object, object]:
    """Load model and tokenizer.

    bits=4  — BitsAndBytesConfig 4-bit NF4 (requires bitsandbytes + CUDA)
    bits=None — bf16, no quantization
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if bits == 4:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Hook construction
# ---------------------------------------------------------------------------

def _resolve_submodule(model, dotted_name: str):
    obj = model
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


def _build_hooks(
    schedule: List[ScheduleEntry],
    vectors: Dict[str, Dict[int, torch.Tensor]],
    probe_traits: List[str],
    probe_layers: List[int],
    token_counter: Dict[str, int],
) -> Tuple[List[Tuple[str, callable]], Dict[int, List[Dict[str, float]]]]:
    """Build all forward hooks and return (hook_list, probe_store).

    hook_list   — list of (layer_dotpath, hook_fn) pairs ready for registration
    probe_store — {layer_int: []} — caller reads this after generation
    """
    # Determine tick layer: highest int across schedule + probe layers
    all_layers = set(e.layer for e in schedule) | set(probe_layers)
    tick_layer = max(all_layers) if all_layers else None

    # Group schedule entries by layer
    by_layer: Dict[int, List[ScheduleEntry]] = {}
    for entry in schedule:
        by_layer.setdefault(entry.layer, []).append(entry)

    # Pre-load probe vectors: {trait: {layer: (v_hat, norm)}}
    probe_vecs: Dict[str, Dict[int, Tuple[torch.Tensor, float]]] = {}
    for trait in probe_traits:
        probe_vecs[trait] = {}
        for pl in probe_layers:
            if trait in vectors and pl in vectors[trait]:
                v = vectors[trait][pl].float()
                norm = v.norm().item()
                probe_vecs[trait][pl] = (v / norm if norm > 0 else v, norm)

    probe_store: Dict[int, List[Dict[str, float]]] = {pl: [] for pl in probe_layers}

    hook_list: List[Tuple[str, callable]] = []

    # Collect all layers that need a hook (steering layers ∪ probe layers)
    all_hook_layers = set(by_layer.keys()) | set(probe_layers)

    for layer_int in sorted(all_hook_layers):
        layer_path = f"model.layers.{layer_int}"
        entries = by_layer.get(layer_int, [])
        is_tick = (layer_int == tick_layer)
        pl = layer_int if layer_int in probe_layers else None

        def _make_hook(entries=entries, pl=pl, is_tick=is_tick, layer_int=layer_int):
            def hook(module, inputs, output):
                t = token_counter["t"]
                hidden = output[0] if isinstance(output, tuple) else output

                # skip prefill pass (seq_len > 1 = entire prompt processed at once)
                if hidden.shape[1] > 1:
                    return output

                # ── steering ────────────────────────────────────────────────
                active = [
                    e for e in entries
                    if e.start <= t and (e.end is None or t < e.end)
                ]
                if active:
                    additive_sum = None
                    for e in active:
                        if e.layer not in vectors.get(e.trait, {}):
                            continue
                        v = vectors[e.trait][e.layer].float().to(hidden.device).to(hidden.dtype)
                        if e.mode == "additive":
                            delta = e.coeff * v
                            additive_sum = delta if additive_sum is None else additive_sum + delta
                        elif e.mode == "rotation":
                            # ORBIT-style norm-preserving rotation. e.coeff = θ (radians).
                            import math
                            v_norm = v.norm()
                            if v_norm < 1e-8:
                                continue
                            v_hat = v / v_norm
                            h_norm = hidden.norm(dim=-1, keepdim=True)
                            h_hat = hidden / h_norm.clamp(min=1e-8)
                            dot = (h_hat * v_hat).sum(dim=-1, keepdim=True)
                            v_perp = v_hat - dot * h_hat
                            v_perp_n = v_perp.norm(dim=-1, keepdim=True)
                            v_perp_hat = v_perp / v_perp_n.clamp(min=1e-8)
                            cos_t = math.cos(float(e.coeff))
                            sin_t = math.sin(float(e.coeff))
                            h_rotated = h_norm * (cos_t * h_hat + sin_t * v_perp_hat)
                            hidden = torch.where(v_perp_n < 1e-6, hidden, h_rotated)
                        else:  # adaptive
                            norm = v.norm()
                            if norm == 0:
                                continue
                            v_hat = v / norm
                            target = e.coeff * norm
                            proj = (hidden * v_hat).sum(dim=-1, keepdim=True)
                            correction = target - proj
                            correction = (
                                correction.clamp(min=0) if e.coeff >= 0
                                else correction.clamp(max=0)
                            )
                            hidden = hidden + correction * v_hat

                    if additive_sum is not None:
                        hidden = hidden + additive_sum.to(hidden.device).to(hidden.dtype)

                # ── probing ──────────────────────────────────────────────────
                if pl is not None:
                    scores: Dict[str, float] = {}
                    for trait, layer_vecs in probe_vecs.items():
                        if pl in layer_vecs:
                            v_hat, _norm = layer_vecs[pl]
                            v_hat = v_hat.to(hidden.device).to(hidden.dtype)
                            h_mean = hidden.float().mean(dim=1).squeeze(0)
                            # projection onto unit steering direction (not double-normalised)
                            score = float((h_mean * v_hat.float()).sum())
                            scores[trait] = score
                    probe_store[pl].append(scores)

                # ── tick ─────────────────────────────────────────────────────
                if is_tick:
                    token_counter["t"] += 1

                if isinstance(output, tuple):
                    return (hidden,) + tuple(output[1:])
                return hidden

            return hook

        hook_list.append((layer_path, _make_hook()))

    return hook_list, probe_store


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation(
    model,
    tokenizer,
    prompt: str,
    schedule: List[ScheduleEntry],
    vectors: Dict[str, Dict[int, torch.Tensor]],
    probe_traits: List[str],
    probe_layers: List[int],
    max_new_tokens: int = 256,
    enable_thinking: bool = False,
) -> Tuple[str, List[str], Dict[int, List[Dict[str, float]]]]:
    """Run steered generation and return (text, token_strings, probe_data)."""
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if enable_thinking:
        text_input += "<think>\n"

    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    token_counter = {"t": 0}
    hook_list, probe_store = _build_hooks(
        schedule, vectors, probe_traits, probe_layers, token_counter
    )

    # Register hooks
    handles = []
    for layer_path, hook_fn in hook_list:
        module = _resolve_submodule(model, layer_path)
        handles.append(module.register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in handles:
            h.remove()

    gen_ids = out[0][input_len:]

    if enable_thinking:
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        eos = tokenizer.eos_token or ""
        text = text.rstrip().removesuffix(eos).rstrip()
    else:
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    token_strings = [tokenizer.decode([tid]) for tid in gen_ids.tolist()]

    return text, token_strings, probe_store


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Categorical palette — adjacent-pairs CVD-safe
_CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300"]


def plot_probe(
    token_strings: List[str],
    probe_data: Dict[int, List[Dict[str, float]]],
    schedule: List[ScheduleEntry],
    layer: int,
    traits: List[str],
) -> "matplotlib.figure.Figure":
    """Return a matplotlib Figure showing per-token probe scores for one layer.

    - One line per trait in `traits`
    - Translucent shaded bands for each ScheduleEntry window (end=None → last token)
    - Horizontal dashed line at y=0
    - X-axis labeled with token strings every 10 tokens
    """
    import matplotlib.pyplot as plt

    per_layer = probe_data.get(layer, [])
    n = len(per_layer)
    xs = list(range(n))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#f5f4f0")
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Trait lines
    for ci, trait in enumerate(traits):
        ys = [step.get(trait, float("nan")) for step in per_layer]
        ax.plot(xs, ys, color=_CAT[ci % len(_CAT)], lw=1.8, label=trait)

    # Shaded bands for each schedule entry
    band_colors = _CAT[len(traits) % len(_CAT):]  # offset to avoid clashing with trait lines
    for bi, entry in enumerate(schedule):
        x0 = entry.start
        x1 = (entry.end - 1) if entry.end is not None else (n - 1)
        x1 = min(x1, n - 1)
        color = band_colors[bi % len(band_colors)]
        ax.axvspan(x0, x1, alpha=0.12, color=color,
                   label=f"{entry.trait} α={entry.coeff}")

    ax.axhline(0, color="#c8c6be", lw=0.8, ls="--")

    # Auto-scale y with 20% padding so lines are never flush with the edge
    ax.autoscale(axis="y")
    ylo, yhi = ax.get_ylim()
    margin = max(abs(yhi - ylo) * 0.2, 1.0)   # at least ±1 so empty plots aren't zero-height
    ax.set_ylim(ylo - margin, yhi + margin)

    # X-axis: show token string every ~20 positions
    tick_step = max(1, n // 20)
    tick_positions = list(range(0, n, tick_step))
    tick_labels = [token_strings[i] if i < len(token_strings) else "" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Generated token index", fontsize=9)
    ax.set_ylabel("h · v̂  (projection onto steering direction)", fontsize=9)
    ax.set_title(f"Persona probe — layer {layer}", fontsize=11)
    ax.legend(loc="upper right", fontsize=7.5, ncol=3, framealpha=0.85, frameon=True)
    ax.grid(axis="y", alpha=0.25, lw=0.6)

    fig.tight_layout()
    return fig
