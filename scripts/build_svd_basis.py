"""Offline SVD persona-space basis builder.

Reads PersVecGen *_raw.pt files (standard + optional amoral traits),
deduplicates near-identical CAA directions, and runs per-layer SVD to
produce a single basis file consumed by SVDPersonaProbe and SVDSteering.

Usage:
    python scripts/build_svd_basis.py \\
        --std-dir   data/vector_extraction/persona/qwen3-4b/bf16 \\
        --amoral-dir data/vector_extraction/persona/qwen3-4b-amoral-roleplay \\
        --hook attn --model qwen3-4b --out data/svd_basis/qwen3-4b_attn.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# PersVecGen hook-key mapping
_HOOK_KEYS: dict[str, str] = {
    "attn": "attn_delta",
    "mlp": "mlp_delta",
    "residual": "residual",
}


def load_caa_vectors(
    raw_dir: Path,
    slug: str,
    hook: str,
    all_layers: List[int],
) -> Dict[int, torch.Tensor]:
    """Load mean-diff CAA vector per layer for one trait.

    Returns {layer: Tensor[d]} for every layer in all_layers.
    Raises FileNotFoundError if the *_raw.pt file is absent.
    """
    hook_key = _HOOK_KEYS[hook]
    path = raw_dir / f"{slug}_raw.pt"
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    return {
        l: (
            raw["positive"][l][hook_key].float().mean(0)
            - raw["negative"][l][hook_key].float().mean(0)
        )
        for l in all_layers
        if l in raw["positive"] and hook_key in raw["positive"][l]
    }


def _cosine_sim_matrix(M: torch.Tensor) -> torch.Tensor:
    """Compute N×N pairwise cosine similarity for rows of M [N, d]."""
    norms = M.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = M / norms
    return normed @ normed.T


def greedy_merge(sims: torch.Tensor, threshold: float = 0.90) -> Dict[int, int]:
    """Return {idx: canonical_idx} via greedy union-find merge.

    Pairs with cosine_sim > threshold are merged (highest-sim first).
    Each idx maps to its canonical representative (lowest-index root).
    """
    N = sims.size(0)
    parent = list(range(N))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Collect upper-triangle pairs sorted by similarity descending
    pairs: List[Tuple[float, int, int]] = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((sims[i, j].item(), i, j))
    pairs.sort(key=lambda x: -x[0])

    for sim, i, j in pairs:
        if sim < threshold:
            break
        ri, rj = _find(i), _find(j)
        if ri != rj:
            # Merge higher root into lower root (lower index = canonical)
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    return {i: _find(i) for i in range(N)}


def apply_merge(
    parent: Dict[int, int],
    slugs: List[str],
    M: Dict[int, torch.Tensor],
) -> Tuple[List[str], Dict[int, torch.Tensor]]:
    """Apply merge map to CAA matrix M (keyed by layer).

    M[layer] is expected as a dict value of shape [N, d] if given as
    {layer: Tensor} or built externally. When called from run_build,
    M is passed as {layer: stacked_Tensor[N, d]}.

    Returns:
        dedup_slugs: list of canonical trait names (length N_dedup <= N)
        M_dedup: {layer: Tensor[N_dedup, d]}, merged rows averaged
    """
    N = len(slugs)
    # Determine canonical order (preserve first occurrence of each canonical)
    seen: Dict[int, int] = {}
    canonical_order: List[int] = []
    for i in range(N):
        c = parent[i]
        if c not in seen:
            seen[c] = len(canonical_order)
            canonical_order.append(c)

    dedup_slugs = [slugs[c] for c in canonical_order]
    n_dedup = len(canonical_order)

    M_dedup: Dict[int, torch.Tensor] = {}
    for layer, mat in M.items():
        # mat: [N, d]
        d = mat.shape[1]
        result = torch.zeros(n_dedup, d)
        counts = torch.zeros(n_dedup)
        for i in range(N):
            pos = seen[parent[i]]
            result[pos] += mat[i]
            counts[pos] += 1
        M_dedup[layer] = result / counts.unsqueeze(1).clamp(min=1.0)

    return dedup_slugs, M_dedup


def build_svd(
    M_dedup: Dict[int, torch.Tensor],
    rank: Optional[int],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int]]:
    """Run SVD on each layer's dedup CAA matrix.

    Returns:
        Vk:        {layer: Tensor[k, d]} — top-k right singular vectors (orthonormal rows)
        C:         {layer: Tensor[N_dedup, k]} — coordinate matrix = M_dedup @ Vk.T
        sigma:     {layer: Tensor[k]} — top-k singular values
        eff_rank:  {layer: int} — effective rank k used
    """
    Vk: Dict[int, torch.Tensor] = {}
    C: Dict[int, torch.Tensor] = {}
    sigma_k: Dict[int, torch.Tensor] = {}
    eff_rank: Dict[int, int] = {}
    for layer, mat in M_dedup.items():
        _, s, Vt = torch.linalg.svd(mat.float(), full_matrices=False)
        if rank is not None:
            k = min(rank, s.numel())
        else:
            p = s / s.sum().clamp(min=1e-12)
            k = max(1, round(1.0 / (p ** 2).sum().item()))
            k = min(k, s.numel())
        Vk[layer] = Vt[:k]            # [k, d]
        C[layer] = mat @ Vt[:k].T     # [N_dedup, k]
        sigma_k[layer] = s[:k]
        eff_rank[layer] = k
    return Vk, C, sigma_k, eff_rank


def run_build(
    std_dir: Path,
    amoral_dir: Optional[Path],
    hook: str,
    model: str,
    ref_layer: int,
    merge_threshold: float,
    rank: Optional[int],
    out_path: Path,
) -> None:
    """Full pipeline: load → dedup → SVD → save."""
    if hook not in _HOOK_KEYS:
        raise ValueError(f"hook must be one of {list(_HOOK_KEYS)}; got {hook!r}")

    # ── 1. Discover all slugs and their directories ──────────────────────────
    slug_dirs: Dict[str, Path] = {}
    for raw_pt in sorted(std_dir.glob("*_raw.pt")):
        slug = raw_pt.stem[:-4]  # strip _raw
        slug_dirs[slug] = std_dir
    if amoral_dir is not None:
        for raw_pt in sorted(amoral_dir.glob("*_raw.pt")):
            slug = raw_pt.stem[:-4]
            slug_dirs[slug] = amoral_dir
    all_slugs = list(slug_dirs.keys())
    if not all_slugs:
        raise RuntimeError(f"No *_raw.pt files found under {std_dir}")

    # ── 2. Determine all_layers from first raw file ──────────────────────────
    first_path = slug_dirs[all_slugs[0]] / f"{all_slugs[0]}_raw.pt"
    probe_raw = torch.load(str(first_path), map_location="cpu", weights_only=False)
    all_layers: List[int] = list(probe_raw["layers"])

    # ── 3. Load CAA vectors for every trait × layer ──────────────────────────
    caa: Dict[str, Dict[int, torch.Tensor]] = {}
    skipped = []
    for slug in all_slugs:
        try:
            caa[slug] = load_caa_vectors(slug_dirs[slug], slug, hook, all_layers)
        except (KeyError, FileNotFoundError):
            skipped.append(slug)
    if skipped:
        print(f"[warn] skipped {len(skipped)} traits missing hook '{hook}': {skipped}")
    present_slugs = [s for s in all_slugs if s in caa]

    # ── 4. Stack into M[layer] = Tensor[N, d] ────────────────────────────────
    M: Dict[int, torch.Tensor] = {}
    for layer in all_layers:
        rows = [caa[s][layer] for s in present_slugs if layer in caa[s]]
        if len(rows) < len(present_slugs):
            continue  # skip layers where not all traits have data
        M[layer] = torch.stack(rows, dim=0)  # [N, d]

    # ── 5. Dedup at ref_layer ─────────────────────────────────────────────────
    if ref_layer not in M:
        ref_layer = list(M.keys())[len(M) // 2]
        print(f"[warn] ref_layer not in data; using layer {ref_layer} instead")
    sims = _cosine_sim_matrix(M[ref_layer])
    parent = greedy_merge(sims, threshold=merge_threshold)
    dedup_slugs, M_dedup = apply_merge(parent, present_slugs, M)

    # Build merge_map: {orig_slug: canonical_slug}
    seen_canon: Dict[int, str] = {}
    for i in range(len(present_slugs)):
        c = parent[i]
        if c not in seen_canon:
            seen_canon[c] = present_slugs[c]
    merge_map = {present_slugs[i]: seen_canon[parent[i]] for i in range(len(present_slugs))}

    # ── 6. SVD per layer ─────────────────────────────────────────────────────
    Vk, C, sigma, eff_rank = build_svd(M_dedup, rank=rank)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "hook":           hook,
        "model":          model,
        "slugs":          dedup_slugs,
        "all_slugs":      all_slugs,
        "merge_map":      merge_map,
        "Vk":             Vk,
        "C":              C,
        "sigma":          sigma,
        "effective_rank": eff_rank,
    }, str(out_path))
    n_merged = len(present_slugs) - len(dedup_slugs)
    print(f"[done] {len(present_slugs)} traits → {len(dedup_slugs)} after dedup "
          f"({n_merged} merged); basis saved to {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Build SVD persona basis from PersVecGen raw .pt files.")
    p.add_argument("--std-dir",  required=True, type=Path)
    p.add_argument("--amoral-dir", default=None, type=Path)
    p.add_argument("--hook",   default="attn", choices=list(_HOOK_KEYS))
    p.add_argument("--model",  required=True)
    p.add_argument("--ref-layer",       default=18, type=int)
    p.add_argument("--merge-threshold", default=0.90, type=float)
    p.add_argument("--rank",   default=None, type=int)
    p.add_argument("--out",    required=True, type=Path)
    args = p.parse_args(argv)
    run_build(
        std_dir=args.std_dir,
        amoral_dir=args.amoral_dir,
        hook=args.hook,
        model=args.model,
        ref_layer=args.ref_layer,
        merge_threshold=args.merge_threshold,
        rank=args.rank,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
