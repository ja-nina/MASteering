"""Deep analysis of no-persona LoRA adapters.

Two analysis modes:

  Weight analysis (CPU, fast, covers all LoRAs):
    For each o_proj LoRA, compute ΔW = (α/r)·B·A per layer.
    Reports: Frobenius norm, effective rank, top SVD write-direction vs
    SVD trait basis (M_dedup) cosine-sim, cross-LoRA weight similarity.

  Activation analysis (GPU, optional):
    Run IPD prompts through base+LoRA, hook residual stream per layer.
    Reports: per-layer activation shift magnitude, projection onto trait
    directions — the empirical "amplification curve".

Usage
-----
    # Weight-only (CPU):
    python scripts/analyze_lora_deep.py \
        --lora-root /scratch/.../loras_ipd_nudge_v3_nopersona \
        --basis     data/svd_basis/qwen3-4b-attn-with-amoral.pt \
        --output-dir results/lora_deep

    # With activation analysis (GPU):
    python scripts/analyze_lora_deep.py \
        --lora-root /scratch/.../loras_ipd_nudge_v3_nopersona \
        --basis     data/svd_basis/qwen3-4b-attn-with-amoral.pt \
        --output-dir results/lora_deep \
        --model Qwen/Qwen3-4B --n-prompts 10 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── helpers ───────────────────────────────────────────────────────────────────

def effective_rank(singular_values: torch.Tensor) -> float:
    """Effective rank = exp(entropy of normalised singular value distribution)."""
    sv = singular_values.float()
    sv = sv[sv > 1e-10]
    if sv.numel() == 0:
        return 0.0
    p = sv / sv.sum()
    return math.exp(-(p * p.log()).sum().item())


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-8)).item()


def all_trait_sims(
    write_dir: torch.Tensor,   # [d]
    M: torch.Tensor,           # [N_traits, d]
) -> torch.Tensor:
    """Cosine-sim between write_dir and every trait direction. Returns [N_traits]."""
    wd = write_dir.float()
    wd_norm = wd / wd.norm().clamp(min=1e-8)
    M_norm = M.float()
    M_norms = M_norm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return (M_norm / M_norms) @ wd_norm   # [N_traits]


def top_trait_alignments(
    write_dir: torch.Tensor,       # [d]
    M: torch.Tensor,               # [N_traits, d]
    slugs: List[str],
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """Return top-k (slug, cosine_sim) sorted by |sim| descending."""
    sims = all_trait_sims(write_dir, M)
    vals, idxs = sims.abs().topk(min(top_k, len(slugs)))
    result = [(slugs[idx], round(sims[idx].item(), 4)) for idx in idxs.tolist()]
    return sorted(result, key=lambda x: -abs(x[1]))


# ── adapter loading ───────────────────────────────────────────────────────────

def find_latest_checkpoint_adapter_b(lora_dir: Path) -> Path:
    checkpoints = sorted(lora_dir.glob("checkpoint_step*"), key=lambda p: p.name)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_step* in {lora_dir}")
    adapter_dir = checkpoints[-1] / "adapter_b" / "adapter_b"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter_b/adapter_b not found under {checkpoints[-1]}")
    return adapter_dir


def load_lora_weights(adapter_dir: Path) -> Tuple[dict, dict]:
    """Return (adapter_config, {key: tensor}) for all lora_A/lora_B tensors."""
    import json as _json
    cfg_path = adapter_dir / "adapter_config.json"
    with open(cfg_path) as f:
        cfg = _json.load(f)

    # Try safetensors first, fall back to pytorch bin
    st_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if st_path.exists():
        from safetensors.torch import load_file
        weights = load_file(str(st_path), device="cpu")
    elif bin_path.exists():
        weights = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter weights in {adapter_dir}")

    return cfg, weights


def extract_delta_w_per_layer(cfg: dict, weights: dict) -> Dict[int, torch.Tensor]:
    """Compute ΔW = (alpha/rank)·lora_B·lora_A for every LoRA layer.

    Returns {layer_idx: ΔW tensor [d_out, d_in]}.
    Only processes o_proj (the only target module in no-persona LoRAs).
    """
    rank  = cfg.get("r", 16)
    alpha = cfg.get("lora_alpha", 32)
    scale = alpha / rank

    # Group weights by layer index
    # key format: base_model.model.layers.{i}.self_attn.o_proj.lora_{A,B}.weight
    layers_A: Dict[int, torch.Tensor] = {}
    layers_B: Dict[int, torch.Tensor] = {}

    for key, tensor in weights.items():
        m = re.search(r'layers\.(\d+)\.self_attn\.o_proj\.lora_([AB])\.weight', key)
        if not m:
            continue
        layer_idx = int(m.group(1))
        ab = m.group(2)
        if ab == "A":
            layers_A[layer_idx] = tensor.float()
        else:
            layers_B[layer_idx] = tensor.float()

    delta_w: Dict[int, torch.Tensor] = {}
    for layer_idx in sorted(set(layers_A) & set(layers_B)):
        A = layers_A[layer_idx]   # [rank, d_in]
        B = layers_B[layer_idx]   # [d_out, rank]
        delta_w[layer_idx] = scale * (B @ A)   # [d_out, d_in]

    return delta_w


# ── weight analysis for one LoRA ──────────────────────────────────────────────

def analyse_weights(
    lora_dir: Path,
    basis: Optional[dict],
) -> dict:
    """Full weight-space analysis for a single no-persona LoRA checkpoint."""
    trait_slug = lora_dir.name   # e.g. ipdNV3np_max_conscientious

    try:
        adapter_dir = find_latest_checkpoint_adapter_b(lora_dir)
    except FileNotFoundError as e:
        return {"lora_dir": str(lora_dir), "error": str(e)}

    try:
        cfg, weights = load_lora_weights(adapter_dir)
    except Exception as e:
        return {"lora_dir": str(lora_dir), "error": f"load failed: {e}"}

    delta_w = extract_delta_w_per_layer(cfg, weights)
    if not delta_w:
        return {"lora_dir": str(lora_dir), "error": "no o_proj LoRA layers found"}

    # Flatten all ΔW for cross-LoRA weight similarity (concat over layers)
    flat_parts = []
    layer_stats = []

    slugs   = basis["slugs"]        if basis else []
    M_dedup = basis.get("M_dedup")  if basis else None

    # Accumulate full trait-alignment vectors across layers for profile similarity
    profile_accum: Optional[torch.Tensor] = None
    profile_layers = 0

    # Parse own target trait once.
    # Dir names use underscores (over_pathologizing); basis slugs use hyphens.
    parts = lora_dir.name.split("_", 2)   # ipdNV3np, direction, trait
    own_trait_raw = parts[2] if len(parts) == 3 else None
    own_trait = own_trait_raw.replace("_", "-") if own_trait_raw else None

    for layer_idx in sorted(delta_w):
        dw = delta_w[layer_idx]   # [d_out, d_in]

        try:
            U, S, Vt = torch.linalg.svd(dw, full_matrices=False)
        except Exception:
            U, S, Vt = None, None, None

        frob_norm = dw.norm("fro").item()
        eff_rank  = effective_rank(S) if S is not None else None
        top_svs   = S[:8].tolist() if S is not None else []

        top_traits = []
        target_alignment = None
        if S is not None and U is not None and M_dedup and layer_idx in M_dedup:
            M = M_dedup[layer_idx].float()
            write_dir = U[:, 0]   # [d_model] — top write direction into residual stream

            # Full alignment vector (all N_traits) — accumulated for profile similarity
            full_sims = all_trait_sims(write_dir, M)   # [N_traits]
            if profile_accum is None:
                profile_accum = full_sims.clone()
            else:
                profile_accum += full_sims
            profile_layers += 1

            top_traits = top_trait_alignments(write_dir, M, slugs, top_k=10)
            # Look up target trait directly from full_sims (avoids top-k miss)
            if own_trait and own_trait in slugs:
                idx = slugs.index(own_trait)
                target_alignment = round(full_sims[idx].item(), 4)

        flat_parts.append(dw.flatten())
        layer_stats.append({
            "layer":               layer_idx,
            "frob_norm":           round(frob_norm, 5),
            "effective_rank":      round(eff_rank, 3) if eff_rank is not None else None,
            "top_singular_values": [round(v, 5) for v in top_svs],
            "top_write_traits":    top_traits,
            "target_alignment":    round(target_alignment, 4) if target_alignment is not None else None,
        })

    total_frob = math.sqrt(sum(s["frob_norm"] ** 2 for s in layer_stats))

    # Mean trait-alignment profile across layers [N_traits]
    trait_profile = None
    if profile_accum is not None and profile_layers > 0:
        trait_profile = (profile_accum / profile_layers).tolist()

    return {
        "lora_dir":        str(lora_dir),
        "adapter_dir":     str(adapter_dir),
        "rank":            cfg.get("r"),
        "alpha":           cfg.get("lora_alpha"),
        "n_layers":        len(layer_stats),
        "total_frob_norm": round(total_frob, 5),
        "layer_stats":     layer_stats,
        "_flat_dw":        torch.cat(flat_parts) if flat_parts else None,
        "_trait_profile":  torch.tensor(trait_profile) if trait_profile else None,
    }


# ── activation analysis for one LoRA ─────────────────────────────────────────

SYSTEM = ("You are a strategic game player. Think carefully before acting — "
          "thorough reasoning leads to better decisions.")

USER_TEMPLATES = [
    "Round {r}. Your opponent cooperated last round. What do you do?",
    "Round {r}. Your opponent defected last round. What do you do?",
    "Round {r}. First round — no history yet. What do you do?",
    "Round {r}. Scores — you: {s1}, opponent: {s2}. What is your move?",
    "Round {r}. Opponent has defected twice in a row. Your move?",
    "Round {r}. You both cooperated last 3 rounds. Continue?",
    "Round {r}. Final round. What is your strategy?",
    "Round {r}. Mixed: 3 cooperate, 2 defect from opponent. What now?",
]


def make_prompts(n: int, tokenizer) -> List[str]:
    prompts = []
    for i in range(n):
        tmpl = USER_TEMPLATES[i % len(USER_TEMPLATES)]
        user = tmpl.format(r=i + 1, s1=i * 3, s2=i * 2 + 1)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user",   "content": user}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    return prompts


def _hook_residual_stream(model, n_layers: int):
    """Register forward hooks on every residual stream output.

    Returns (handles, activations_dict).
    activations_dict[layer_idx] is populated after each forward pass.
    """
    acts: Dict[int, torch.Tensor] = {}
    handles = []

    def _make_hook(idx):
        def _h(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            acts[idx] = h[:, -1, :].detach().float().cpu()  # last token, [1, d]
        return _h

    for i in range(n_layers):
        layer = model.model.layers[i]
        handles.append(layer.register_forward_hook(_make_hook(i)))

    return handles, acts


@torch.no_grad()
def analyse_activations(
    lora_dir: Path,
    base_model,
    tokenizer,
    basis: dict,
    n_prompts: int,
    device: str,
) -> dict:
    """Measure per-layer residual stream amplification toward trait directions."""
    from peft import PeftModel

    try:
        adapter_dir = find_latest_checkpoint_adapter_b(lora_dir)
    except FileNotFoundError as e:
        return {"error": str(e)}

    M_dedup = basis.get("M_dedup", {})
    slugs   = basis["slugs"]
    n_layers = base_model.config.num_hidden_layers

    prompts = make_prompts(n_prompts, tokenizer)
    lora_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    lora_model.eval()

    # ── collect base and LoRA activations per prompt ─────────────────────────
    handles, acts = _hook_residual_stream(lora_model, n_layers)
    layer_shift_sums: Dict[int, torch.Tensor] = {}   # accumulated ΔActivation
    layer_shift_norms: Dict[int, List[float]] = {}

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(device)

        # Base pass
        lora_model.base_model.disable_adapter_layers()
        lora_model(**enc)
        base_acts = {l: acts[l].clone() for l in range(n_layers)}

        # LoRA pass
        lora_model.base_model.enable_adapter_layers()
        lora_model(**enc)
        lora_acts = {l: acts[l].clone() for l in range(n_layers)}

        for l in range(n_layers):
            shift = (lora_acts[l] - base_acts[l]).squeeze(0)   # [d]
            layer_shift_sums.setdefault(l, torch.zeros_like(shift))
            layer_shift_sums[l] += shift
            layer_shift_norms.setdefault(l, [])
            layer_shift_norms[l].append(shift.norm().item())

    for h in handles:
        h.remove()
    restored_base = lora_model.unload()
    torch.cuda.empty_cache()

    # ── project mean shift onto trait directions ───────────────────────────
    layer_results = []
    for l in sorted(layer_shift_sums):
        mean_shift = layer_shift_sums[l] / n_prompts   # [d]
        shift_norm = float(torch.tensor(layer_shift_norms[l]).mean())

        top_traits = []
        target_alignment = None
        if l in M_dedup:
            M = M_dedup[l].float()
            full_sims = all_trait_sims(mean_shift, M)
            top_traits = top_trait_alignments(mean_shift, M, slugs, top_k=10)
            parts = lora_dir.name.split("_", 2)
            if len(parts) == 3:
                own_trait = parts[2].replace("_", "-")
                if own_trait in slugs:
                    target_alignment = round(full_sims[slugs.index(own_trait)].item(), 4)

        layer_results.append({
            "layer":             l,
            "mean_shift_norm":   round(shift_norm, 5),
            "top_shift_traits":  top_traits,
            "target_alignment":  target_alignment,
        })

    return {
        "n_prompts":    n_prompts,
        "layer_stats":  layer_results,
        "_base_model":  restored_base,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep LoRA analysis (no-persona)")
    parser.add_argument("--lora-root", type=Path, required=True,
                        help="Root dir containing ipdNV3np_* subdirectories")
    parser.add_argument("--basis", type=Path, default=None,
                        help="SVD basis .pt file (M_dedup key required for alignment)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=None,
                        help="Base model ID; enables activation analysis when provided")
    parser.add_argument("--n-prompts", type=int, default=10,
                        help="Prompts per LoRA for activation analysis (0 = skip)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── load SVD basis ────────────────────────────────────────────────────────
    basis = None
    if args.basis and args.basis.exists():
        basis = torch.load(str(args.basis), map_location="cpu", weights_only=False)
        print(f"Basis loaded: {len(basis['slugs'])} traits, "
              f"hook={basis.get('hook', '?')}, "
              f"layers={sorted(basis.get('M_dedup', basis.get('Vk', {})).keys())[:5]}...")
    else:
        print("No basis file — skipping trait alignment.")

    # ── discover LoRA directories ─────────────────────────────────────────────
    lora_dirs = sorted([p for p in args.lora_root.iterdir()
                        if p.is_dir() and p.name.startswith("ipdNV3np_")])
    print(f"Found {len(lora_dirs)} no-persona LoRA directories")

    # ── Phase 1: weight analysis (CPU, all LoRAs) ─────────────────────────────
    print("\n── Phase 1: weight analysis ─────────────────────────────────────")
    weight_results = []
    flat_dws: Dict[str, torch.Tensor] = {}   # for cross-LoRA similarity

    for i, lora_dir in enumerate(lora_dirs):
        print(f"  [{i+1:3d}/{len(lora_dirs)}] {lora_dir.name}", flush=True)
        r = analyse_weights(lora_dir, basis)
        dw = r.pop("_flat_dw", None)
        if dw is not None:
            flat_dws[lora_dir.name] = dw
        weight_results.append(r)

    # ── cross-LoRA similarity matrices ───────────────────────────────────────
    sim_matrix: Optional[List[List[float]]] = None
    profile_sim_matrix: Optional[List[List[float]]] = None
    names_for_sim: List[str] = []

    # Collect trait profiles alongside flat dws
    flat_profiles: Dict[str, torch.Tensor] = {}
    for r in weight_results:
        if "_trait_profile" in r and r["_trait_profile"] is not None:
            flat_profiles[Path(r["lora_dir"]).name] = r.pop("_trait_profile")

    if len(flat_dws) > 1:
        print("  Computing cross-LoRA weight similarity matrix...")
        keys = sorted(flat_dws)
        vecs = torch.stack([flat_dws[k] for k in keys])
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sim_matrix = ((vecs / norms) @ (vecs / norms).T).tolist()
        names_for_sim = keys

    if len(flat_profiles) > 1:
        print("  Computing cross-LoRA trait-profile similarity matrix...")
        pkeys = sorted(flat_profiles)
        pvecs = torch.stack([flat_profiles[k] for k in pkeys])
        pnorms = pvecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        profile_sim_matrix = ((pvecs / pnorms) @ (pvecs / pnorms).T).tolist()
        # pkeys should match names_for_sim if both ran; store separately if different
        if pkeys != names_for_sim:
            names_for_sim = pkeys  # fall back to profile keys

    # ── Phase 2: activation analysis (GPU, optional) ─────────────────────────
    activation_results = []
    if args.model and args.n_prompts > 0 and basis:
        print("\n── Phase 2: activation analysis ────────────────────────────────")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        print(f"  Loading base model {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype_map[args.dtype],
            device_map=args.device,
            trust_remote_code=True,
        )
        base_model.eval()

        for i, lora_dir in enumerate(lora_dirs):
            print(f"  [{i+1:3d}/{len(lora_dirs)}] {lora_dir.name}", flush=True)
            r = analyse_activations(lora_dir, base_model, tokenizer,
                                    basis, args.n_prompts, args.device)
            if "_base_model" in r:
                base_model = r.pop("_base_model")
            activation_results.append({"lora_dir": str(lora_dir), **r})
    else:
        print("\nSkipping activation analysis (pass --model and --n-prompts > 0 to enable)")

    # ── summary statistics ────────────────────────────────────────────────────
    valid_w = [r for r in weight_results if "error" not in r]
    print(f"\n── Summary ─────────────────────────────────────────────────────")
    print(f"  Weight analysis : {len(valid_w)}/{len(weight_results)} successful")
    if valid_w:
        norms = [r["total_frob_norm"] for r in valid_w]
        ranks = [s["effective_rank"] for r in valid_w
                 for s in r["layer_stats"] if s["effective_rank"] is not None]
        targets = [s["target_alignment"] for r in valid_w
                   for s in r["layer_stats"] if s["target_alignment"] is not None]
        print(f"  Total ||ΔW||_F  : {min(norms):.4f} – {max(norms):.4f}  "
              f"(mean {sum(norms)/len(norms):.4f})")
        if ranks:
            print(f"  Effective rank  : {min(ranks):.2f} – {max(ranks):.2f}  "
                  f"(mean {sum(ranks)/len(ranks):.2f})")
        if targets:
            print(f"  Target alignment: {min(targets):.4f} – {max(targets):.4f}  "
                  f"(mean {sum(targets)/len(targets):.4f})")

    # ── save outputs ──────────────────────────────────────────────────────────
    # Strip non-serialisable / already-consumed items
    for r in weight_results:
        r.pop("_flat_dw", None)
        r.pop("_trait_profile", None)

    combined = {
        "weight_analysis":              weight_results,
        "activation_analysis":          activation_results,
        "cross_lora_weight_similarity": {
            "names": names_for_sim,
            "matrix": sim_matrix,
        } if sim_matrix else None,
        "cross_lora_profile_similarity": {
            "names": names_for_sim,
            "matrix": profile_sim_matrix,
        } if profile_sim_matrix else None,
    }
    out_json = args.output_dir / "lora_deep_analysis.json"
    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nFull results → {out_json}")

    # ── generate heatmap data for visualisation ───────────────────────────────
    _write_heatmap_data(weight_results, args.output_dir)
    _write_html_report(combined, args.output_dir)
    print(f"HTML report  → {args.output_dir / 'report.html'}")


def _write_heatmap_data(weight_results: list, out_dir: Path):
    """Save compact per-layer norm matrix (trait × layer) for the heatmap."""
    valid = [r for r in weight_results if "error" not in r and r.get("layer_stats")]
    if not valid:
        return
    n_layers = max(s["layer"] for r in valid for s in r["layer_stats"]) + 1
    rows = []
    for r in valid:
        norm_by_layer = {s["layer"]: s["frob_norm"] for s in r["layer_stats"]}
        rows.append({
            "name":   Path(r["lora_dir"]).name,
            "norms":  [round(norm_by_layer.get(l, 0.0), 5) for l in range(n_layers)],
        })
    with open(out_dir / "heatmap_data.json", "w") as f:
        json.dump({"n_layers": n_layers, "rows": rows}, f)


def _write_html_report(data: dict, out_dir: Path):
    """Write a self-contained HTML report with interactive heatmap and charts."""
    weight  = data["weight_analysis"]
    sim     = data.get("cross_lora_weight_similarity")
    psim    = data.get("cross_lora_profile_similarity")
    act     = data.get("activation_analysis", [])

    valid_w = [r for r in weight if "error" not in r and r.get("layer_stats")]

    # Collect per-layer norm table
    if valid_w:
        n_layers = max(s["layer"] for r in valid_w for s in r["layer_stats"]) + 1
        trait_names = [Path(r["lora_dir"]).name.replace("ipdNV3np_", "") for r in valid_w]
        heatmap_matrix = []
        for r in valid_w:
            norm_by_layer = {s["layer"]: s["frob_norm"] for s in r["layer_stats"]}
            heatmap_matrix.append([norm_by_layer.get(l, 0.0) for l in range(n_layers)])
        # Target alignment summary
        align_summary = []
        for r in valid_w:
            vals = [s["target_alignment"] for s in r["layer_stats"]
                    if s["target_alignment"] is not None]
            name = Path(r["lora_dir"]).name.replace("ipdNV3np_", "")
            mean_align = sum(vals) / len(vals) if vals else 0.0
            align_summary.append({"name": name, "mean_target_alignment": round(mean_align, 4),
                                   "total_frob_norm": r["total_frob_norm"]})
        align_summary.sort(key=lambda x: -x["mean_target_alignment"])
    else:
        n_layers, trait_names, heatmap_matrix, align_summary = 0, [], [], []

    sim_names        = sim["names"]    if sim  else []
    sim_matrix       = sim["matrix"]   if sim  else []
    psim_names       = psim["names"]   if psim else []
    psim_matrix      = psim["matrix"]  if psim else []

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LoRA Deep Analysis</title>
<style>
:root {{
  --bg: #f8f9fa; --surface: #fff; --border: #dee2e6;
  --text: #212529; --muted: #6c757d;
  --accent: #2563eb; --accent2: #dc2626;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8;
    --accent: #60a5fa; --accent2: #f87171;
  }}
}}
body {{ margin: 0; font-family: system-ui, sans-serif; background: var(--bg);
        color: var(--text); font-size: 14px; }}
h1 {{ padding: 24px 32px 0; margin: 0; font-size: 20px; }}
h2 {{ font-size: 15px; margin: 0 0 12px; color: var(--text); }}
.section {{ margin: 24px 32px; padding: 20px; background: var(--surface);
            border: 1px solid var(--border); border-radius: 8px; }}
canvas {{ display: block; max-width: 100%; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
@media (max-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ padding: 4px 8px; text-align: right; border-bottom: 1px solid var(--border); }}
th {{ font-weight: 600; text-align: center; color: var(--muted); }}
td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>No-Persona LoRA Deep Analysis</h1>
<p style="padding: 4px 32px; color: var(--muted)">
  {len(valid_w)} LoRAs analysed &nbsp;·&nbsp;
  Weight-space (ΔW = B·A per o_proj layer) &nbsp;·&nbsp;
  {'Activation analysis included' if act else 'Activation analysis not run'}
</p>

<div class="section">
  <h2>||ΔW||_F per layer — heatmap (trait × layer)</h2>
  <p style="font-size:12px;color:var(--muted)">
    Colour intensity = Frobenius norm of the LoRA weight delta at that layer.
    Reveals which layers each trait modifies most.
  </p>
  <div class="scroll"><canvas id="heatmap"></canvas></div>
</div>

<div class="grid2" style="margin:0 32px 24px">
  <div class="section" style="margin:0">
    <h2>Mean target-trait write-direction alignment</h2>
    <p style="font-size:12px;color:var(--muted)">
      Cosine-sim between the LoRA's top SVD write direction (U[:,0] of ΔW)
      and the target trait's CAA direction, averaged across layers.
      High = LoRA writes directly toward the intended trait.
    </p>
    <canvas id="alignChart" height="400"></canvas>
  </div>
  <div class="section" style="margin:0">
    <h2>Total ||ΔW||_F vs KL divergence</h2>
    <p style="font-size:12px;color:var(--muted)">
      Does more weight change → more distributional drift?
    </p>
    <canvas id="scatterChart"></canvas>
  </div>
</div>

<div class="grid2" style="margin:0 32px 24px">
  {f'''<div class="section" style="margin:0">
    <h2>Weight similarity (ΔW cosine-sim)</h2>
    <p style="font-size:12px;color:var(--muted)">
      Pairwise cosine-sim of flattened ΔW vectors. High = two LoRAs modify
      the same weight directions. Expect max/min pairs to be anti-correlated.
    </p>
    <div class="scroll"><canvas id="simMatrix"></canvas></div>
  </div>''' if sim_matrix else '<div></div>'}
  {f'''<div class="section" style="margin:0">
    <h2>Trait-profile similarity</h2>
    <p style="font-size:12px;color:var(--muted)">
      Pairwise cosine-sim of each LoRA's mean trait-alignment vector (cosine-sim
      with every CAA direction, averaged across layers). High = two LoRAs push the
      model toward the same traits behaviourally.
    </p>
    <div class="scroll"><canvas id="psimMatrix"></canvas></div>
  </div>''' if psim_matrix else '<div></div>'}
</div>

<div class="section">
  <h2>Top write-direction trait alignments per LoRA</h2>
  <div class="scroll">
  <table id="alignTable">
    <thead><tr>
      <th>LoRA</th><th>Total ||ΔW||</th><th>Mean target alignment</th>
    </tr></thead>
    <tbody>
{''.join(f"<tr><td>{r['name']}</td><td>{r['total_frob_norm']:.4f}</td>"
         f"<td>{r['mean_target_alignment']:+.4f}</td></tr>"
         for r in align_summary[:60])}
    </tbody>
  </table>
  </div>
</div>

<script>
const HEATMAP_TRAITS = {json.dumps(trait_names)};
const HEATMAP_MATRIX = {json.dumps(heatmap_matrix)};
const N_LAYERS = {n_layers};
const ALIGN_DATA = {json.dumps(align_summary)};
const SIM_NAMES = {json.dumps(sim_names)};
const SIM_MATRIX = {json.dumps(sim_matrix if sim_matrix else [])};
const PSIM_NAMES = {json.dumps(psim_names)};
const PSIM_MATRIX = {json.dumps(psim_matrix if psim_matrix else [])};

// ── colour helpers ────────────────────────────────────────────────────────────
function lerp(a, b, t) {{ return a + (b - a) * t; }}
function normColor(v, lo, hi) {{
  const t = Math.max(0, Math.min(1, (v - lo) / Math.max(hi - lo, 1e-9)));
  const r = Math.round(lerp(240, 37,  t));
  const g = Math.round(lerp(240, 99,  t));
  const b = Math.round(lerp(240, 235, t));
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function simColor(v) {{
  // -1 red → 0 white → +1 blue
  if (v >= 0) {{
    const t = v;
    return `rgb(${{Math.round(lerp(240,37,t))}},${{Math.round(lerp(240,99,t))}},${{Math.round(lerp(240,235,t))}})`;
  }} else {{
    const t = -v;
    return `rgb(${{Math.round(lerp(240,220,t))}},${{Math.round(lerp(240,38,t))}},${{Math.round(lerp(240,38,t))}})`;
  }}
}}
function getTheme() {{
  const dt = document.documentElement.getAttribute('data-theme');
  if (dt === 'dark') return 'dark';
  if (dt === 'light') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}}
function textColor() {{ return getTheme() === 'dark' ? '#e2e8f0' : '#212529'; }}

// ── heatmap ───────────────────────────────────────────────────────────────────
(function drawHeatmap() {{
  if (!HEATMAP_MATRIX.length) return;
  const canvas = document.getElementById('heatmap');
  const CELL = 8, LABEL_W = 180, LABEL_H = 30;
  canvas.width  = LABEL_W + N_LAYERS * CELL;
  canvas.height = LABEL_H + HEATMAP_MATRIX.length * CELL;
  const ctx = canvas.getContext('2d');
  const allVals = HEATMAP_MATRIX.flat();
  const lo = Math.min(...allVals), hi = Math.max(...allVals);
  ctx.font = '9px system-ui';
  ctx.fillStyle = textColor();
  // layer labels
  for (let l = 0; l < N_LAYERS; l += 4) {{
    ctx.fillText(l, LABEL_W + l * CELL, LABEL_H - 2);
  }}
  // rows
  HEATMAP_MATRIX.forEach((row, ti) => {{
    ctx.font = '9px system-ui';
    ctx.fillStyle = textColor();
    const label = HEATMAP_TRAITS[ti] || '';
    ctx.fillText(label.slice(-22), 0, LABEL_H + ti * CELL + CELL - 1);
    row.forEach((v, li) => {{
      ctx.fillStyle = normColor(v, lo, hi);
      ctx.fillRect(LABEL_W + li * CELL, LABEL_H + ti * CELL, CELL - 1, CELL - 1);
    }});
  }});
}})();

// ── alignment bar chart ───────────────────────────────────────────────────────
(function drawAlignChart() {{
  if (!ALIGN_DATA.length) return;
  const canvas = document.getElementById('alignChart');
  const top40 = ALIGN_DATA.slice(0, 40);
  const BAR_H = 16, LABEL_W = 160, PAD = 8;
  canvas.height = PAD + top40.length * (BAR_H + 2) + PAD;
  canvas.width  = LABEL_W + 300;
  const ctx = canvas.getContext('2d');
  const maxV = Math.max(...top40.map(d => Math.abs(d.mean_target_alignment)), 0.01);
  top40.forEach((d, i) => {{
    const y = PAD + i * (BAR_H + 2);
    ctx.font = '10px system-ui';
    ctx.fillStyle = textColor();
    ctx.fillText(d.name.slice(-22), 0, y + BAR_H - 3);
    const w = Math.abs(d.mean_target_alignment) / maxV * 260;
    ctx.fillStyle = d.mean_target_alignment >= 0 ? '#2563eb' : '#dc2626';
    ctx.fillRect(LABEL_W, y, w, BAR_H - 2);
    ctx.fillStyle = textColor();
    ctx.fillText(d.mean_target_alignment.toFixed(3), LABEL_W + w + 4, y + BAR_H - 3);
  }});
}})();

// ── scatter chart ─────────────────────────────────────────────────────────────
(function drawScatter() {{
  if (!ALIGN_DATA.length) return;
  const canvas = document.getElementById('scatterChart');
  canvas.width = 400; canvas.height = 320;
  const ctx = canvas.getContext('2d');
  const PAD = 40;
  const W = canvas.width - PAD * 2, H = canvas.height - PAD * 2;
  const norms = ALIGN_DATA.map(d => d.total_frob_norm);
  const aligns = ALIGN_DATA.map(d => d.mean_target_alignment);
  const xlo = Math.min(...norms), xhi = Math.max(...norms);
  const ylo = Math.min(...aligns), yhi = Math.max(...aligns);
  ctx.strokeStyle = getTheme() === 'dark' ? '#334155' : '#dee2e6';
  ctx.beginPath(); ctx.moveTo(PAD, PAD); ctx.lineTo(PAD, PAD + H); ctx.lineTo(PAD + W, PAD + H); ctx.stroke();
  ctx.font = '10px system-ui'; ctx.fillStyle = textColor();
  ctx.fillText('||ΔW||_F', PAD + W / 2 - 20, canvas.height - 4);
  ALIGN_DATA.forEach(d => {{
    const x = PAD + (d.total_frob_norm - xlo) / Math.max(xhi - xlo, 1e-9) * W;
    const y = PAD + H - (d.mean_target_alignment - ylo) / Math.max(yhi - ylo, 1e-9) * H;
    const isMax = d.name.startsWith('max_');
    ctx.fillStyle = isMax ? '#2563eb88' : '#dc262688';
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
  }});
  // legend
  ctx.fillStyle = '#2563eb'; ctx.fillRect(PAD, PAD, 8, 8);
  ctx.fillStyle = textColor(); ctx.fillText('max', PAD + 10, PAD + 8);
  ctx.fillStyle = '#dc2626'; ctx.fillRect(PAD + 40, PAD, 8, 8);
  ctx.fillStyle = textColor(); ctx.fillText('min', PAD + 50, PAD + 8);
}})();

// ── similarity matrix renderer (reusable) ────────────────────────────────────
function drawSimMatrixOnCanvas(canvasId, names, matrix) {{
  if (!matrix.length) return;
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const N = names.length;
  const CELL = 6, LABEL = 130;
  canvas.width  = LABEL + N * CELL;
  canvas.height = LABEL + N * CELL;
  const ctx = canvas.getContext('2d');
  ctx.font = '7px system-ui';
  names.forEach((name, i) => {{
    const label = name.replace('ipdNV3np_', '').slice(-20);
    ctx.save();
    ctx.translate(LABEL + i * CELL + CELL, LABEL - 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = textColor();
    ctx.fillText(label, 0, 0);
    ctx.restore();
    ctx.fillStyle = textColor();
    ctx.fillText(label, 0, LABEL + i * CELL + CELL - 1);
  }});
  matrix.forEach((row, ri) => {{
    row.forEach((v, ci) => {{
      ctx.fillStyle = simColor(v);
      ctx.fillRect(LABEL + ci * CELL, LABEL + ri * CELL, CELL - 0.5, CELL - 0.5);
    }});
  }});
}}
drawSimMatrixOnCanvas('simMatrix',  SIM_NAMES,  SIM_MATRIX);
drawSimMatrixOnCanvas('psimMatrix', PSIM_NAMES, PSIM_MATRIX);
</script>
</body>
</html>"""

    (out_dir / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
