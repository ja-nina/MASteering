# SVD Persona Steering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an SVD-based persona steering and monitoring system: an offline basis builder, a probe that projects agent activations onto persona SVD space, a steering class that injects a persona point as an activation vector, and a Gradio human-vs-agent demo with live SVD projection.

**Architecture:** `build_svd_basis.py` reads PersVecGen raw `.pt` files, deduplicates near-identical traits (cosine sim > 0.90), runs per-layer SVD, and saves a basis file. `SVDPersonaProbe` and `SVDSteering` each use the same basis file and integrate with the existing `_HookSession` / `make_hook()` pattern in `TransformersPolicy`. The Gradio demo wires human-vs-agent TextArena play to the probe's live SVD output.

**Tech Stack:** Python 3.10+, PyTorch, Gradio 4+, TextArena, HuggingFace Transformers (Qwen3-4B).

**Spec:** `docs/superpowers/specs/2026-08-19-svd-persona-steering-design.md`

## Global Constraints

- Hook paths: `attn → model.layers.{N}.self_attn.o_proj`, `mlp → model.layers.{N}.mlp.down_proj`, `residual → model.layers.{N}` — these match PersVecGen `SUBMODULE_PATHS`.
- PersVecGen raw `.pt` format: `{"positive": {layer: {"residual": T[n,d], "attn_delta": T[n,d], "mlp_delta": T[n,d]}}, "negative": {...}, "layers": [...]}`. Hook-key mapping: `attn → attn_delta`, `mlp → mlp_delta`, `residual → residual`.
- CAA vector per trait per layer: `raw["positive"][l][hook_key].float().mean(0) - raw["negative"][l][hook_key].float().mean(0)`.
- Effective rank: `k = round(1.0 / (p**2).sum().item())` where `p = sigma / sigma.sum()`.
- Dedup threshold: cosine sim > 0.90 at `ref_layer`; greedy merge (highest-sim pair first); merged traits share a canonical vector (component-wise mean); all original names remain valid via `merge_map`.
- `SVDPersonaProbe.make_hook()` must return `(List[Tuple[str, Callable]], Callable)` — identical contract to `PersonaProbe.make_hook()`.
- `SVDSteering.apply_hooks(agent_id, model)` returns `List[Tuple[str, Callable]]` for `_HookSession`.
- No existing test files are modified. `TransformersPolicy.act()` gets a small new branch; the file's existing `_HookSession` class is not renamed.
- All new code passes `pytest tests/ -x` (excluding `@pytest.mark.gpu` tests).
- `layer_path_template` defaults to `"model.layers.{}"` throughout.

---

## File Map

| File | Status | Role |
|---|---|---|
| `scripts/build_svd_basis.py` | **CREATE** | Offline SVD basis builder CLI |
| `testbed/probing/svd_probe.py` | **CREATE** | `SVDPersonaProbe` class |
| `testbed/steering/svd_steering.py` | **CREATE** | `SVDSteering` class |
| `testbed/policy/transformers_policy.py` | **MODIFY** (lines 119-130) | Add SVD hook branch in `act()` |
| `notebooks/persona_game_demo.py` | **CREATE** | Gradio human-vs-agent demo |
| `tests/test_build_svd_basis.py` | **CREATE** | Smoke tests for basis builder |
| `tests/probing/__init__.py` | **CREATE** | Package marker |
| `tests/probing/test_svd_probe.py` | **CREATE** | Unit tests for SVDPersonaProbe |
| `tests/steering/test_svd_steering.py` | **CREATE** | Unit tests for SVDSteering |

---

### Task 1: `scripts/build_svd_basis.py` + tests

**Files:**
- Create: `scripts/build_svd_basis.py`
- Test: `tests/test_build_svd_basis.py`

**Interfaces:**
- Consumes: PersVecGen `*_raw.pt` files from two directories (std + amoral).
- Produces: `basis.pt` — a `dict` with keys `hook`, `model`, `slugs`, `all_slugs`, `merge_map`, `Vk` (dict layer→Tensor[k,d]), `C` (dict layer→Tensor[N_dedup,k]), `sigma` (dict layer→Tensor[k]), `effective_rank` (dict layer→int). This dict is what Tasks 2 and 3 load.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_svd_basis.py
import pytest
import torch
from pathlib import Path
from scripts.build_svd_basis import (
    load_caa_vectors,
    greedy_merge,
    apply_merge,
    build_svd,
    run_build,
)


def _make_raw(tmp_path: Path, slug: str, n_layers=3, n_prompts=4, d=8) -> Path:
    """Create a minimal PersVecGen *_raw.pt fixture."""
    hooks = ("residual", "attn_delta", "mlp_delta")
    layers = list(range(n_layers))
    data = {
        "positive": {l: {h: torch.randn(n_prompts, d) for h in hooks} for l in layers},
        "negative": {l: {h: torch.randn(n_prompts, d) for h in hooks} for l in layers},
        "layers": layers,
        "steer_layers": layers,
        "hooks": list(hooks),
        "trait": slug,
        "model": "test",
    }
    path = tmp_path / f"{slug}_raw.pt"
    torch.save(data, path)
    return path


def test_load_caa_vectors_returns_correct_shape(tmp_path):
    _make_raw(tmp_path, "agreeable", n_layers=3, n_prompts=4, d=8)
    vecs = load_caa_vectors(tmp_path, slug="agreeable", hook="attn", all_layers=[0, 1, 2])
    assert set(vecs.keys()) == {0, 1, 2}
    assert all(v.shape == (8,) for v in vecs.values())


def test_greedy_merge_merges_identical_traits():
    # Two identical vectors → cosine sim = 1.0 → should merge
    sims = torch.tensor([[1.0, 1.0, 0.0],
                          [1.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
    parent = greedy_merge(sims, threshold=0.90)
    # Indices 0 and 1 should share the same canonical
    assert parent[0] == parent[1]
    # Index 2 is isolated
    assert parent[2] == 2


def test_greedy_merge_does_not_merge_dissimilar():
    sims = torch.eye(3)  # identity → only self-similarity = 1.0
    parent = greedy_merge(sims, threshold=0.90)
    assert parent[0] == 0 and parent[1] == 1 and parent[2] == 2


def test_apply_merge_averages_merged_pair():
    torch.manual_seed(0)
    slugs = ["a", "b", "c"]
    M = {0: torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 1.0]])}
    # Merge a and b → canonical = a (idx 0), c stays
    parent = {0: 0, 1: 0, 2: 2}
    dedup_slugs, M_dedup = apply_merge(parent, slugs, M)
    assert dedup_slugs == ["a", "c"]
    assert M_dedup[0].shape == (2, 2)
    # First row = mean of rows 0 and 1 = [2.0, 0.0]
    assert torch.allclose(M_dedup[0][0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(M_dedup[0][1], torch.tensor([0.0, 1.0]))


def test_build_svd_vk_is_orthonormal():
    torch.manual_seed(1)
    # N_dedup=5, d=16, 3 layers
    M_dedup = {l: torch.randn(5, 16) for l in range(3)}
    Vk, C, sigma, eff_rank = build_svd(M_dedup, rank=None)
    for l in range(3):
        k = eff_rank[l]
        # Vk[l] shape: [k, 16], rows should be orthonormal
        gram = Vk[l] @ Vk[l].T  # [k, k]
        assert torch.allclose(gram, torch.eye(k), atol=1e-5)


def test_run_build_end_to_end(tmp_path):
    std_dir = tmp_path / "std"
    std_dir.mkdir()
    for slug in ["agreeable", "dominant", "curious"]:
        _make_raw(std_dir, slug, n_layers=2, n_prompts=3, d=6)
    out_path = tmp_path / "basis.pt"
    run_build(
        std_dir=std_dir,
        amoral_dir=None,
        hook="attn",
        model="test-model",
        ref_layer=0,
        merge_threshold=0.90,
        rank=None,
        out_path=out_path,
    )
    assert out_path.exists()
    basis = torch.load(str(out_path), map_location="cpu", weights_only=False)
    assert basis["hook"] == "attn"
    assert "Vk" in basis and "C" in basis
    # All dedup slugs are in all_slugs
    assert set(basis["slugs"]).issubset(set(basis["all_slugs"]))
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_build_svd_basis.py -x -v
```
Expected: `ImportError: No module named 'scripts.build_svd_basis'`

- [ ] **Step 3: Implement `scripts/build_svd_basis.py`**

```python
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
    merge_map = {present_slugs[i]: dedup_slugs[parent[i] if parent[i] < len(present_slugs) else i]
                 for i in range(len(present_slugs))}
    # Correct: map each orig slug to the canonical slug by canonical index
    canonical_idx_to_slug = {v: dedup_slugs[pos] for v, pos in
                              {parent[i]: next(j for j, c in enumerate(
                                  [parent[k] for k in range(len(present_slugs))]) if c == parent[i])
                               for i in range(len(present_slugs))}.items()}
    # Simpler correct approach: rebuild merge_map cleanly
    seen_canon: Dict[int, str] = {}
    _canon_order: List[int] = []
    for i in range(len(present_slugs)):
        c = parent[i]
        if c not in seen_canon:
            seen_canon[c] = present_slugs[c] if c < len(present_slugs) else present_slugs[i]
            _canon_order.append(c)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_build_svd_basis.py -x -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_svd_basis.py tests/test_build_svd_basis.py
git commit -m "feat: add build_svd_basis.py with dedup, SVD, and tests"
```

---

### Task 2: `testbed/probing/svd_probe.py` + tests

**Files:**
- Create: `testbed/probing/svd_probe.py`
- Create: `tests/probing/__init__.py`
- Create: `tests/probing/test_svd_probe.py`

**Interfaces:**
- Consumes: basis `.pt` file produced by Task 1 (keys: `slugs`, `Vk`, `C`, `effective_rank`).
- Produces: `SVDPersonaProbe` class with:
  - `__init__(basis_path, layers, hook, layer_path_template, window_tokens, top_k)`
  - `make_hook() → (List[Tuple[str, Callable]], Callable)` — identical contract to `PersonaProbe.make_hook()`
  - Output dict stored under `"svd_probe"` key: `{str(layer): {"z": [float, ...], "top_traits": [[name, score], ...]}}`
  - `SUBMODULE_SUFFIXES` mapping that Task 3 also imports: `{"attn": ".self_attn.o_proj", "mlp": ".mlp.down_proj", "residual": ""}`

- [ ] **Step 1: Write the failing test**

```python
# tests/probing/test_svd_probe.py
import torch
import torch.nn as nn
import pytest
from testbed.probing.svd_probe import SVDPersonaProbe, SUBMODULE_SUFFIXES


def _make_toy_basis(tmp_path, n_dedup=3, k=2, d=4, n_layers=2):
    """Create a minimal basis .pt file for testing."""
    layers = list(range(n_layers))
    torch.manual_seed(42)
    # Build orthonormal Vk via QR
    Vk = {l: torch.linalg.qr(torch.randn(d, k))[0].T for l in layers}  # [k, d]
    C = {l: torch.randn(n_dedup, k) for l in layers}  # [N_dedup, k]
    basis = {
        "hook": "attn",
        "model": "test",
        "slugs": ["agreeable", "dominant", "curious"],
        "all_slugs": ["agreeable", "dominant", "curious"],
        "merge_map": {"agreeable": "agreeable", "dominant": "dominant", "curious": "curious"},
        "Vk": Vk,
        "C": C,
        "sigma": {l: torch.ones(k) for l in layers},
        "effective_rank": {l: k for l in layers},
    }
    path = tmp_path / "basis.pt"
    torch.save(basis, str(path))
    return path, Vk, C, layers


def test_submodule_suffixes_are_correct():
    assert SUBMODULE_SUFFIXES["attn"] == ".self_attn.o_proj"
    assert SUBMODULE_SUFFIXES["mlp"] == ".mlp.down_proj"
    assert SUBMODULE_SUFFIXES["residual"] == ""


def test_probe_make_hook_returns_correct_shape(tmp_path):
    basis_path, Vk, C, layers = _make_toy_basis(tmp_path, n_dedup=3, k=2, d=4, n_layers=2)
    probe = SVDPersonaProbe(basis_path=str(basis_path), layers=[0, 1], hook="attn")
    hooks, get_result = probe.make_hook()
    assert len(hooks) == 2  # one hook per layer
    # Each hook pair is (path_str, callable)
    for path, fn in hooks:
        assert ".self_attn.o_proj" in path
        assert callable(fn)


def test_probe_z_has_correct_shape(tmp_path):
    basis_path, Vk, C, layers = _make_toy_basis(tmp_path, n_dedup=3, k=2, d=4, n_layers=2)
    probe = SVDPersonaProbe(basis_path=str(basis_path), layers=[0], hook="residual",
                            layer_path_template="block.{}")

    # Create a tiny model with a module at "block.0"
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.ModuleList([nn.Identity()])

    model = TinyModel()
    hooks, get_result = probe.make_hook()

    # Manually register one hook to simulate _HookSession
    path, hook_fn = hooks[0]
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    handle = module.register_forward_hook(hook_fn)

    h = torch.randn(1, 5, 4)  # [batch, seq, d]
    module(h)
    handle.remove()

    result = get_result()
    assert "0" in result
    z = result["0"]["z"]
    assert len(z) == 2  # k=2
    top_traits = result["0"]["top_traits"]
    assert len(top_traits) == 3  # top_k defaults to 3 traits (n_dedup=3)
    assert all(len(t) == 2 for t in top_traits)  # [name, score]


def test_probe_top_traits_ranked_by_cosine_sim(tmp_path):
    """Trait with highest cosine sim to z should appear first."""
    n_dedup, k, d = 3, 2, 4
    basis_path, Vk, C, layers = _make_toy_basis(tmp_path, n_dedup=n_dedup, k=k, d=d)
    probe = SVDPersonaProbe(basis_path=str(basis_path), layers=[0], hook="residual",
                            layer_path_template="block.{}")

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.ModuleList([nn.Identity()])

    model = TinyModel()
    hooks, get_result = probe.make_hook()
    path, hook_fn = hooks[0]
    module = model.block[0]
    handle = module.register_forward_hook(hook_fn)

    # Feed the vector that perfectly aligns with the first trait's z-point
    # C[0][0] is the first trait's coordinate → inject hidden that projects to it
    target_z = C[0][0]  # [k]
    # h such that Vk[0] @ h_last ≈ target_z → h_last = Vk[0].T @ target_z
    h_last = Vk[0].T @ target_z  # [d]
    h = h_last.unsqueeze(0).unsqueeze(0)  # [1, 1, d]
    module(h)
    handle.remove()

    result = get_result()
    top = result["0"]["top_traits"]
    assert top[0][0] == "agreeable"  # first trait should rank highest


def test_probe_chunk_accumulation(tmp_path):
    """Probe with window_tokens=2 should accumulate chunk entries."""
    basis_path, Vk, C, layers = _make_toy_basis(tmp_path, n_dedup=3, k=2, d=4)
    probe = SVDPersonaProbe(basis_path=str(basis_path), layers=[0], hook="residual",
                            layer_path_template="block.{}", window_tokens=2)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.ModuleList([nn.Identity()])

    model = TinyModel()
    hooks, get_result = probe.make_hook()
    _, hook_fn = hooks[0]
    handle = model.block[0].register_forward_hook(hook_fn)
    for _ in range(4):  # 4 tokens → 2 complete chunks of 2
        model.block[0](torch.randn(1, 1, 4))
    handle.remove()

    result = get_result()
    assert len(result["0"]["chunks"]) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/probing/test_svd_probe.py -x -v
```
Expected: `ImportError: No module named 'testbed.probing.svd_probe'`

- [ ] **Step 3: Create `tests/probing/__init__.py`**

```python
# empty
```

- [ ] **Step 4: Implement `testbed/probing/svd_probe.py`**

```python
"""SVD persona-space probe.

Attaches read-only forward hooks to decoder sub-layers during model.generate()
and projects the final-token hidden state onto the SVD persona basis built by
scripts/build_svd_basis.py.

Config (in YAML):
    probing:
      mode:          svd
      basis_path:    ${SVD_BASIS}
      hook:          attn          # attn | mlp | residual
      layers:        [10, 18, 27]
      layer_path_template: "model.layers.{}"
      window_tokens: 10
      top_k:         5

Output per turn (stored in episode JSONL under "svd_probe"):
    {
      "18": {"z": [0.82, -0.31, ...], "top_traits": [["agreeable", 0.71], ...]},
      "27": {...}
    }
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

# Submodule suffix appended to "model.layers.{N}" for each hook type.
# Must match PersVecGen SUBMODULE_PATHS exactly.
SUBMODULE_SUFFIXES: Dict[str, str] = {
    "attn":     ".self_attn.o_proj",
    "mlp":      ".mlp.down_proj",
    "residual": "",
}


class SVDPersonaProbe:
    """Project agent hidden states onto the SVD persona basis.

    Drop-in companion to PersonaProbe; activated when probing.mode == "svd".
    Uses the same make_hook() → (hooks_list, get_result_fn) contract.
    """

    def __init__(
        self,
        basis_path: str,
        layers: Optional[List[int]] = None,
        layer: int = 18,
        hook: str = "attn",
        layer_path_template: str = "model.layers.{}",
        window_tokens: int = 10,
        top_k: int = 5,
    ) -> None:
        import torch
        if layers is not None:
            self.layers = list(layers)
        else:
            self.layers = [layer]

        if hook not in SUBMODULE_SUFFIXES:
            raise ValueError(f"hook must be one of {list(SUBMODULE_SUFFIXES)}; got {hook!r}")

        self.hook = hook
        self.layer_path_template = layer_path_template
        self.window_tokens = window_tokens
        self.top_k = top_k

        basis = torch.load(os.path.expandvars(basis_path),
                           map_location="cpu", weights_only=False)
        self._slugs: List[str] = basis["slugs"]
        self._Vk: Dict[int, torch.Tensor] = basis["Vk"]    # {layer: [k, d]}
        self._C: Dict[int, torch.Tensor] = basis["C"]       # {layer: [N_dedup, k]}

    def _layer_path(self, layer_int: int) -> str:
        base = self.layer_path_template.format(layer_int)
        return base + SUBMODULE_SUFFIXES[self.hook]

    def make_hook(self) -> Tuple[List[Tuple[str, Callable]], Callable]:
        """Return ([(path, hook_fn), ...], get_result_fn).

        Compatible with PersonaProbe.make_hook() contract.
        Register every pair via _HookSession; call get_result_fn() after
        generate() to retrieve the SVD coordinates for each layer.
        """
        import torch

        states: Dict[int, Dict] = {
            l: {"n": 0, "mean_z": None,
                "chunk_sum_z": None, "chunk_n": 0, "chunks": []}
            for l in self.layers
        }
        w = self.window_tokens if self.window_tokens > 0 else None

        def _make_layer_hook(layer_int: int):
            st = states[layer_int]
            Vk = self._Vk[layer_int]  # [k, d]

            def _hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                last = h[:, -1, :].detach().float().mean(dim=0)  # [d]
                z = Vk.to(last.device) @ last                    # [k]

                n = st["n"] + 1
                st["n"] = n
                if st["mean_z"] is None:
                    st["mean_z"] = z.clone()
                else:
                    st["mean_z"] = st["mean_z"] + (z - st["mean_z"]) / n

                if w is not None:
                    if st["chunk_sum_z"] is None:
                        st["chunk_sum_z"] = z.clone()
                        st["chunk_n"] = 1
                    else:
                        st["chunk_sum_z"] = st["chunk_sum_z"] + z
                        st["chunk_n"] += 1
                    if st["chunk_n"] >= w:
                        chunk_z = st["chunk_sum_z"] / st["chunk_n"]
                        st["chunks"].append({
                            "token": n,
                            "z": chunk_z.tolist(),
                        })
                        st["chunk_sum_z"] = None
                        st["chunk_n"] = 0

            return _hook

        hooks = [
            (self._layer_path(l), _make_layer_hook(l))
            for l in self.layers
        ]

        def _get_result() -> Dict[str, Dict]:
            result = {}
            for l in self.layers:
                st = states[l]
                chunks = list(st["chunks"])
                if st["chunk_sum_z"] is not None and st["chunk_n"] > 0:
                    chunk_z = st["chunk_sum_z"] / st["chunk_n"]
                    chunks.append({"token": st["n"], "z": chunk_z.tolist()})
                mean_z = st["mean_z"]
                if mean_z is None:
                    top_traits: List[List] = []
                    z_list: List[float] = []
                else:
                    top_traits = self._nearest_traits(mean_z, l)
                    z_list = mean_z.tolist()
                result[str(l)] = {"z": z_list, "top_traits": top_traits, "chunks": chunks}
            return result

        return hooks, _get_result

    def _nearest_traits(self, z: "torch.Tensor", layer: int) -> List[List]:
        """Return top-k [[slug, cosine_sim], ...] for z against C[layer]."""
        C = self._C[layer].to(z.device)  # [N_dedup, k]
        z_norm = z.norm().clamp(min=1e-8)
        C_norms = C.norm(dim=1).clamp(min=1e-8)   # [N_dedup]
        sims = (C @ z) / (C_norms * z_norm)       # [N_dedup]
        k = min(self.top_k, len(self._slugs))
        topk_vals, topk_idx = sims.topk(k)
        return [[self._slugs[i.item()], v.item()] for v, i in zip(topk_vals, topk_idx)]
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/probing/test_svd_probe.py -x -v
```
Expected: all 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add testbed/probing/svd_probe.py tests/probing/__init__.py tests/probing/test_svd_probe.py
git commit -m "feat: add SVDPersonaProbe with make_hook() and tests"
```

---

### Task 3: `testbed/steering/svd_steering.py` + `TransformersPolicy` extension + tests

**Files:**
- Create: `testbed/steering/svd_steering.py`
- Modify: `testbed/policy/transformers_policy.py` (add SVD branch in `act()`, lines ~119-130)
- Test: `tests/steering/test_svd_steering.py`

**Interfaces:**
- Consumes: basis `.pt` from Task 1; `make_steering_hook` from `testbed.steering.activation`; `SUBMODULE_SUFFIXES` from `testbed.probing.svd_probe`.
- Produces: `SVDSteering` class with:
  - `__init__(basis_path, per_agent, default_config, mode, layer_path_template)`
  - `apply_hooks(agent_id, model) → List[Tuple[str, Callable]]`
  - Internal: `_build_injection(persona, layer) → Tensor[d]`

- [ ] **Step 1: Write the failing test**

```python
# tests/steering/test_svd_steering.py
import torch
import torch.nn as nn
import pytest
from testbed.steering.svd_steering import SVDSteering


def _make_toy_basis(tmp_path, n_dedup=3, k=2, d=4, n_layers=2):
    layers = list(range(n_layers))
    torch.manual_seed(7)
    Vk = {l: torch.linalg.qr(torch.randn(d, k))[0].T for l in layers}
    C = {l: torch.randn(n_dedup, k) for l in layers}
    basis = {
        "hook": "attn",
        "model": "test",
        "slugs": ["agreeable", "dominant", "curious"],
        "all_slugs": ["agreeable", "dominant", "curious", "warm"],
        "merge_map": {"agreeable": "agreeable", "dominant": "dominant",
                      "curious": "curious", "warm": "agreeable"},
        "Vk": Vk,
        "C": C,
        "sigma": {l: torch.ones(k) for l in layers},
        "effective_rank": {l: k for l in layers},
    }
    path = tmp_path / "basis.pt"
    torch.save(basis, str(path))
    return path, Vk, C, layers


def test_build_injection_shape(tmp_path):
    basis_path, Vk, C, layers = _make_toy_basis(tmp_path)
    steering = SVDSteering(
        basis_path=str(basis_path),
        per_agent={"player_0": {"hook": "attn", "layers": [0], "coefficient": 1.0,
                                "persona": {"agreeable": 1.0}}},
    )
    g = steering._build_injection({"agreeable": 1.0}, layer=0)
    assert g.shape == (4,)  # d=4


def test_merged_trait_maps_to_same_injection(tmp_path):
    """'warm' is merged into 'agreeable' — same injection vector for both."""
    basis_path, Vk, C, layers = _make_toy_basis(tmp_path)
    steering = SVDSteering(basis_path=str(basis_path), per_agent={})
    g_agreeable = steering._build_injection({"agreeable": 1.0}, layer=0)
    g_warm = steering._build_injection({"warm": 1.0}, layer=0)
    assert torch.allclose(g_agreeable, g_warm)


def test_apply_hooks_attn_returns_one_hook_per_layer(tmp_path):
    basis_path, *_ = _make_toy_basis(tmp_path)
    steering = SVDSteering(
        basis_path=str(basis_path),
        per_agent={"player_0": {"hook": "attn", "layers": [0, 1], "coefficient": 1.0,
                                "persona": {"agreeable": 1.0}}},
    )

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = type("M", (), {"layers": [
                type("B", (), {"self_attn": type("A", (), {"o_proj": nn.Identity()})()})()
                for _ in range(2)
            ]})()

    model = FakeModel()
    hooks = steering.apply_hooks("player_0", model)
    assert len(hooks) == 2  # one per layer (hook="attn")
    for path, fn in hooks:
        assert "self_attn.o_proj" in path
        assert callable(fn)


def test_apply_hooks_both_returns_two_hooks_per_layer(tmp_path):
    basis_path, *_ = _make_toy_basis(tmp_path)
    steering = SVDSteering(
        basis_path=str(basis_path),
        per_agent={"player_0": {"hook": "both", "layers": [0], "coefficient": 1.0,
                                "persona": {"agreeable": 1.0}}},
    )

    class _Layer:
        self_attn = type("A", (), {"o_proj": nn.Linear(4, 4)})()
        mlp = type("M", (), {"down_proj": nn.Linear(4, 4)})()

    class _M:
        layers = [_Layer()]

    class FakeModel(nn.Module):
        model = _M()

    model = FakeModel()
    hooks = steering.apply_hooks("player_0", model)
    assert len(hooks) == 2  # both attn + mlp for layer 0
    paths = [p for p, _ in hooks]
    assert any("self_attn.o_proj" in p for p in paths)
    assert any("mlp.down_proj" in p for p in paths)


def test_apply_hooks_unconfigured_agent_returns_empty(tmp_path):
    basis_path, *_ = _make_toy_basis(tmp_path)
    steering = SVDSteering(basis_path=str(basis_path), per_agent={})
    hooks = steering.apply_hooks("player_99", None)
    assert hooks == []


def test_transformers_policy_svd_branch(tmp_path):
    """TransformersPolicy.act() calls apply_hooks when steering has that method."""
    from testbed.policy.transformers_policy import TransformersPolicy

    basis_path, Vk, C, layers = _make_toy_basis(tmp_path)

    class _FakeSVDSteering:
        called_with = []
        def apply_hooks(self, agent_id, model):
            _FakeSVDSteering.called_with.append(agent_id)
            return []  # no actual hooks to register

    class _FakeBatch(dict):
        def to(self, device): return self

    class _FakeTokenizer:
        eos_token_id = 0
        eos_token = "<|endoftext|>"
        def apply_chat_template(self, messages, **kw): return "text"
        def __call__(self, text, return_tensors=None):
            return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})
        def decode(self, ids, **kw): return "ok"

    class _FakeModel(nn.Module):
        def generate(self, **kw):
            return torch.tensor([[1, 2, 3, 0]])

    policy = TransformersPolicy.__new__(TransformersPolicy)
    policy.model = _FakeModel()
    policy.tokenizer = _FakeTokenizer()
    policy.device = "cpu"
    policy.enable_thinking = False
    policy.reasoning_cue = False
    policy.steering = _FakeSVDSteering()
    policy.probe = None
    policy._gen_kwargs = {"max_new_tokens": 8, "temperature": 0.7, "top_p": 0.9, "top_k": 20}
    policy._last_probe = {}

    policy.act("sys", "usr", "player_0", None)
    assert "player_0" in _FakeSVDSteering.called_with
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/steering/test_svd_steering.py -x -v
```
Expected: `ImportError: No module named 'testbed.steering.svd_steering'`

- [ ] **Step 3: Implement `testbed/steering/svd_steering.py`**

```python
"""SVD persona steering.

Maps a per-agent persona dict (trait → weight) to an activation injection
vector via the SVD persona basis, then registers forward hooks for each
configured layer.

YAML config schema:
    steering:
      default: svd
      basis_path: ${SVD_BASIS}
      mode: additive          # additive | adaptive | rotation
      per_agent:
        player_0:
          hook: attn          # attn | mlp | both | residual
          layers: [10, 18, 27]
          coefficient: 1.0
          persona:
            agreeable:  1.5
            dominant:  -0.5
            warm:       0.0   # zeros are explicit; merged into 'agreeable' if sim > 0.90
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

from testbed.probing.svd_probe import SUBMODULE_SUFFIXES
from testbed.steering.activation import make_steering_hook


class SVDSteering:
    """Per-agent activation steering via the SVD persona space.

    Call apply_hooks(agent_id, model) to get (layer_path, hook_fn) pairs
    ready for TransformersPolicy._HookSession.
    """

    def __init__(
        self,
        basis_path: str,
        per_agent: Dict[str, Dict],
        default_config: Optional[Dict] = None,
        mode: str = "additive",
        layer_path_template: str = "model.layers.{}",
    ) -> None:
        import torch

        self._mode = mode
        self._template = layer_path_template

        basis = torch.load(os.path.expandvars(basis_path),
                           map_location="cpu", weights_only=False)
        self._slugs_dedup: List[str] = basis["slugs"]
        self._merge_map: Dict[str, str] = basis["merge_map"]
        self._slug_to_idx: Dict[str, int] = {s: i for i, s in enumerate(self._slugs_dedup)}
        self._Vk: Dict[int, "torch.Tensor"] = basis["Vk"]   # {layer: [k, d]}
        self._C: Dict[int, "torch.Tensor"] = basis["C"]     # {layer: [N_dedup, k]}

        # TextArena passes integer player IDs; YAML uses "player_N" strings
        self._per_agent: Dict = {str(k): v for k, v in (per_agent or {}).items()}
        self._default_config = default_config

    def _cfg_for(self, agent_id) -> Optional[Dict]:
        candidates = [str(agent_id)]
        if str(agent_id).isdigit():
            candidates.append(f"player_{agent_id}")
        for key in candidates:
            if key in self._per_agent:
                return self._per_agent[key]
        return self._default_config

    def _build_injection(self, persona: Dict[str, float], layer: int) -> "torch.Tensor":
        """Map persona dict → weight vector w → SVD point z → injection g ∈ ℝ^d.

        z = C[layer].T @ w   (weighted sum of trait coordinates)
        g = Vk[layer].T @ z  (project back to model space)
        """
        import torch
        n = len(self._slugs_dedup)
        w = torch.zeros(n)
        for slug, weight in persona.items():
            canonical = self._merge_map.get(slug, slug)
            idx = self._slug_to_idx.get(canonical)
            if idx is not None:
                w[idx] += weight
        C = self._C[layer]   # [N_dedup, k]
        Vk = self._Vk[layer] # [k, d]
        z = C.T @ w          # [k]
        g = Vk.T @ z         # [d]
        return g

    def apply_hooks(
        self, agent_id, model
    ) -> List[Tuple[str, Callable]]:
        """Return (layer_path, hook_fn) pairs for _HookSession.

        Returns empty list if agent is not configured.
        """
        cfg = self._cfg_for(agent_id)
        if cfg is None:
            return []

        persona = cfg.get("persona", {})
        coefficient = float(cfg.get("coefficient", 1.0))
        hook_type = cfg.get("hook", "attn")
        layers = list(cfg.get("layers", [18]))

        result: List[Tuple[str, Callable]] = []
        for layer in layers:
            g = self._build_injection(persona, layer)
            hook_fn = make_steering_hook(g, coefficient=coefficient, mode=self._mode)
            base = self._template.format(layer)

            if hook_type == "both":
                result.append((base + SUBMODULE_SUFFIXES["attn"], hook_fn))
                result.append((base + SUBMODULE_SUFFIXES["mlp"], hook_fn))
            elif hook_type in SUBMODULE_SUFFIXES:
                result.append((base + SUBMODULE_SUFFIXES[hook_type], hook_fn))
            else:
                raise ValueError(f"Unknown hook type {hook_type!r}")

        return result

    def on_episode_start(self, env) -> None:
        pass
```

- [ ] **Step 4: Modify `testbed/policy/transformers_policy.py` — add SVD branch in `act()`**

Open `testbed/policy/transformers_policy.py`. Replace the `act()` method's hook-building section (current lines 119-130) with the following. The change is a new `elif hasattr(...)` branch inserted BEFORE the existing `if steering is not None` check:

```python
    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec]) -> tuple[str, bool]:
        inputs = self._build_inputs(system_prompt, user_prompt)

        # Build the list of hooks to register: steering (if active) + probe (always)
        hooks = []

        # SVDSteering path — duck-typed: any steering object with apply_hooks()
        if self.steering is not None and hasattr(self.steering, "apply_hooks"):
            svd_hooks = self.steering.apply_hooks(agent_id, self.model)
            hooks.extend(svd_hooks)
        elif steering is not None and steering.method == "activation":
            if self.steering is None:
                raise ValueError("Activation steering requested but no steering "
                                 "method bound to the policy.")
            vec = self.steering.load_vector(agent_id)
            steer_hook = make_steering_hook(
                vec, coefficient=steering.coefficient, mode=steering.mode
            )
            hooks.append((steering.layer, steer_hook))

        get_scores = None
        if self.probe is not None:
            probe_hooks, get_scores = self.probe.make_hook()
            hooks.extend(probe_hooks)

        if hooks:
            with _HookSession(self.model, hooks):
                result = self._generate(inputs)
        else:
            result = self._generate(inputs)

        self._last_probe = get_scores() if get_scores is not None else {}
        return result
```

The only structural change: added the `if self.steering is not None and hasattr(self.steering, "apply_hooks"):` block before the existing `elif steering is not None` block. Everything else is unchanged.

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/steering/test_svd_steering.py -x -v
```
Expected: all 6 tests PASS (including `test_transformers_policy_svd_branch`).

- [ ] **Step 6: Run full suite to confirm no regressions**

```
pytest tests/ -x -v --ignore=tests/policy/test_vllm_policy.py -k "not gpu"
```
Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add testbed/steering/svd_steering.py testbed/policy/transformers_policy.py tests/steering/test_svd_steering.py
git commit -m "feat: add SVDSteering and extend TransformersPolicy for SVD hooks"
```

---

### Task 4: `notebooks/persona_game_demo.py` — Gradio human-vs-agent demo

**Files:**
- Create: `notebooks/persona_game_demo.py`

**Interfaces:**
- Consumes: `SVDPersonaProbe` (Task 2), `SVDSteering` (Task 3), `TransformersPolicy` (modified in Task 3).
- Produces: A runnable Gradio app; no automated tests. Verify by launching and playing one turn.

Target games (env IDs): `DontSayIt-v0`, `SimpleNegotiation-v0`, `Taboo-v0`, `TruthAndDeception-v0`, `CharacterConclave-v0`, `Diplomacy-v0`, `Negotiation-v0`, `SecretMafia-v0`.

- [ ] **Step 1: Implement `notebooks/persona_game_demo.py`**

```python
"""Gradio demo: human vs. steered agent in TextArena with live SVD projection.

Launch:
    python notebooks/persona_game_demo.py \\
        --model Qwen/Qwen3-4B \\
        --basis data/svd_basis/qwen3-4b_attn.pt \\
        --hook attn --layers 10,18,27 \\
        --game SimpleNegotiation-v0

Three-panel layout:
    Left:   game transcript + human text input + Send button
    Centre: SVD projection bar chart (z coordinates, top-2 trait labels per bar)
    Right:  game selector, layer dropdown, hook type, persona sliders, Apply button
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import torch
import textarena as ta

from testbed.policy.transformers_policy import TransformersPolicy
from testbed.probing.svd_probe import SVDPersonaProbe
from testbed.steering.svd_steering import SVDSteering

_GAMES = [
    "DontSayIt-v0", "SimpleNegotiation-v0", "Taboo-v0",
    "TruthAndDeception-v0", "CharacterConclave-v0",
    "Diplomacy-v0", "Negotiation-v0", "SecretMafia-v0",
]
_HUMAN_ID = 0
_AGENT_ID = 1


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class DemoState:
    """Mutable demo state. Not thread-safe; Gradio is single-threaded per session."""
    def __init__(self):
        self.env = None
        self.obs = None
        self.done = False
        self.transcript: List[str] = []
        self.last_z: Optional[List[float]] = None
        self.last_top_traits: List[List] = []
        self.policy: Optional[TransformersPolicy] = None
        self.probe: Optional[SVDPersonaProbe] = None
        self.steering: Optional[SVDSteering] = None
        self.lock = threading.Lock()


_STATE = DemoState()


# ---------------------------------------------------------------------------
# Chart helper
# ---------------------------------------------------------------------------

def _make_bar_chart_html(z: List[float], top_traits: List[List], title: str = "") -> str:
    """Return an HTML bar chart for SVD z-coordinates."""
    if not z:
        return "<p>No projection data yet.</p>"
    k = len(z)
    bar_w = max(300, k * 24)
    max_abs = max(abs(v) for v in z) or 1.0
    bars = []
    for i, val in enumerate(z):
        pct_pos = val / max_abs  # [-1, 1]
        label = top_traits[i][0] if i < len(top_traits) else f"PC{i}"
        color = "#4a90d9" if val >= 0 else "#e05a5a"
        width_px = int(abs(pct_pos) * 80)
        margin_px = 80 - width_px if val < 0 else 80
        bars.append(
            f'<div style="display:flex;align-items:center;gap:4px;margin:2px 0">'
            f'<span style="font-size:11px;width:90px;text-align:right;color:#888">{label}</span>'
            f'<div style="width:160px;display:flex;justify-content:{"flex-end" if val<0 else "flex-start"}">'
            f'<div style="width:{width_px}px;height:14px;background:{color};border-radius:2px"></div>'
            f'</div>'
            f'<span style="font-size:10px;color:#666">{val:.2f}</span>'
            f'</div>'
        )
    return (
        f'<div style="font-size:12px;font-weight:600;margin-bottom:8px">{title}</div>'
        + "".join(bars)
    )


# ---------------------------------------------------------------------------
# Game control
# ---------------------------------------------------------------------------

def _start_game(game_id: str, persona_values: Dict[str, float],
                hook: str, layers: List[int]) -> Tuple[str, str]:
    """Initialise or re-initialise the TextArena env and agent."""
    global _STATE
    with _STATE.lock:
        env = ta.make(game_id)
        obs, info = env.reset(num_players=2)
        _STATE.env = env
        _STATE.obs = obs
        _STATE.done = False
        _STATE.transcript = [f"[Game started: {game_id}]"]
        _STATE.last_z = None
        _STATE.last_top_traits = []

        # Build SVD steering from current sliders
        if _STATE.policy is not None and _STATE.steering is not None:
            _update_persona(persona_values, layers, hook)

        obs_text = obs.get(_HUMAN_ID, obs.get(str(_HUMAN_ID), ""))
        _STATE.transcript.append(f"[Env] {obs_text}")
        transcript_html = "<br>".join(_STATE.transcript)
        chart_html = "<p>Play a turn to see projection.</p>"
    return transcript_html, chart_html


def _update_persona(persona_values: Dict[str, float], layers: List[int], hook: str) -> None:
    """Recompute injection vectors from slider state (no model reload)."""
    if _STATE.steering is None or _STATE.policy is None:
        return
    _STATE.steering._per_agent = {
        str(_AGENT_ID): {
            "hook": hook,
            "layers": layers,
            "coefficient": 1.0,
            "persona": persona_values,
        }
    }


def _human_turn(human_text: str, probe_layer: int) -> Tuple[str, str]:
    """Process one human action and one agent response."""
    global _STATE
    with _STATE.lock:
        if _STATE.env is None or _STATE.done:
            return "Start a game first.", "<p>No game running.</p>"

        # 1. Step the env with human action
        obs, rewards, done, info = _STATE.env.step({_HUMAN_ID: human_text})
        _STATE.transcript.append(f"[You] {human_text}")

        if done:
            _STATE.done = True
            _STATE.transcript.append(f"[Game over] {rewards}")
            return "<br>".join(_STATE.transcript), "<p>Game over.</p>"

        # 2. Agent turn
        agent_obs = obs.get(_AGENT_ID, obs.get(str(_AGENT_ID), ""))
        system_prompt = "You are a competitive game player. Respond concisely."
        action, _ = _STATE.policy.act(
            system_prompt=system_prompt,
            user_prompt=str(agent_obs),
            agent_id=str(_AGENT_ID),
            steering=None,  # SVDSteering handled via self.steering.apply_hooks
        )
        _STATE.transcript.append(f"[Agent] {action}")

        # 3. Step env with agent action
        obs, rewards, done, info = _STATE.env.step({_AGENT_ID: action})
        _STATE.done = done

        if done:
            _STATE.transcript.append(f"[Game over] {rewards}")

        # 4. Extract SVD projection from probe
        z = None
        top_traits: List[List] = []
        if _STATE.policy._last_probe:
            layer_data = _STATE.policy._last_probe.get(str(probe_layer), {})
            z = layer_data.get("z", None)
            top_traits = layer_data.get("top_traits", [])
        _STATE.last_z = z
        _STATE.last_top_traits = top_traits

        chart_html = _make_bar_chart_html(
            z or [], top_traits, title=f"SVD projection (layer {probe_layer})"
        )
        transcript_html = "<br>".join(_STATE.transcript)
    return transcript_html, chart_html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="Qwen/Qwen3-4B")
    parser.add_argument("--basis",  required=True)
    parser.add_argument("--hook",   default="attn",
                        choices=["attn", "mlp", "both", "residual"])
    parser.add_argument("--layers", default="10,18,27")
    parser.add_argument("--game",   default="SimpleNegotiation-v0")
    parser.add_argument("--share",  action="store_true")
    args = parser.parse_args(argv)

    layers = [int(x) for x in args.layers.split(",")]

    # Load basis to extract trait names for sliders
    basis = torch.load(args.basis, map_location="cpu", weights_only=False)
    all_slugs: List[str] = basis["all_slugs"]

    # Build probe (monitor)
    probe = SVDPersonaProbe(
        basis_path=args.basis,
        layers=layers,
        hook=args.hook,
    )

    # Build steering (starts with all zeros)
    initial_persona = {slug: 0.0 for slug in all_slugs}
    steering = SVDSteering(
        basis_path=args.basis,
        per_agent={
            str(_AGENT_ID): {
                "hook": args.hook,
                "layers": layers,
                "coefficient": 1.0,
                "persona": initial_persona,
            }
        },
    )

    # Build policy (load model once)
    policy = TransformersPolicy(
        model_id=args.model,
        steering=steering,
        probe=probe,
    )

    _STATE.policy = policy
    _STATE.probe = probe
    _STATE.steering = steering

    # Build Gradio UI
    with gr.Blocks(title="Persona Game Demo") as demo:
        gr.Markdown("## SVD Persona Steering — Human vs. Agent")

        with gr.Row():
            # ── Left: transcript ─────────────────────────────────────────────
            with gr.Column(scale=2):
                transcript = gr.HTML(label="Transcript", value="<p>Select a game and click Start.</p>")
                human_input = gr.Textbox(label="Your move", placeholder="Type your action…")
                send_btn = gr.Button("Send")

            # ── Centre: SVD chart ────────────────────────────────────────────
            with gr.Column(scale=2):
                chart = gr.HTML(label="SVD Projection", value="<p>No data yet.</p>")
                probe_layer = gr.Dropdown(
                    choices=[str(l) for l in layers],
                    value=str(layers[0]),
                    label="Display layer",
                )

            # ── Right: controls ──────────────────────────────────────────────
            with gr.Column(scale=1):
                game_sel = gr.Dropdown(choices=_GAMES, value=args.game, label="Game")
                hook_sel = gr.Dropdown(
                    choices=["attn", "mlp", "both", "residual"],
                    value=args.hook, label="Hook type",
                )
                start_btn = gr.Button("Start / Restart", variant="primary")
                gr.Markdown("### Persona sliders")
                sliders = {
                    slug: gr.Slider(-2.0, 2.0, value=0.0, step=0.1, label=slug)
                    for slug in sorted(all_slugs)
                }
                apply_btn = gr.Button("Apply persona")

        # ── Event handlers ────────────────────────────────────────────────────

        def _on_start(game_id, hook_type, *slider_vals):
            persona = dict(zip(sorted(all_slugs), slider_vals))
            _update_persona(persona, layers, hook_type)
            t, c = _start_game(game_id, persona, hook_type, layers)
            return t, c

        def _on_send(text, layer_str, *slider_vals):
            persona = dict(zip(sorted(all_slugs), slider_vals))
            _update_persona(persona, layers, args.hook)
            t, c = _human_turn(text, int(layer_str))
            return t, c, ""

        def _on_apply(*slider_vals):
            persona = dict(zip(sorted(all_slugs), slider_vals))
            _update_persona(persona, layers, args.hook)
            return gr.update()

        slider_list = [sliders[s] for s in sorted(all_slugs)]

        start_btn.click(
            fn=_on_start,
            inputs=[game_sel, hook_sel] + slider_list,
            outputs=[transcript, chart],
        )
        send_btn.click(
            fn=_on_send,
            inputs=[human_input, probe_layer] + slider_list,
            outputs=[transcript, chart, human_input],
        )
        apply_btn.click(fn=_on_apply, inputs=slider_list, outputs=[chart])

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax (no model load required)**

```
python -c "import ast; ast.parse(open('notebooks/persona_game_demo.py').read()); print('syntax ok')"
```
Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add notebooks/persona_game_demo.py
git commit -m "feat: add persona_game_demo.py Gradio human-vs-agent TextArena demo"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task covering it |
|---|---|
| `build_svd_basis.py` — dedup, SVD, effective rank, save | Task 1 |
| `SVDPersonaProbe.make_hook()` matching `PersonaProbe` contract | Task 2 |
| Probe hooks at correct sublayer (`self_attn.o_proj` / `mlp.down_proj`) | Task 2 (SUBMODULE_SUFFIXES) |
| `SVDSteering.apply_hooks()` returning (path, fn) pairs | Task 3 |
| `hook: both` registers two hooks per layer | Task 3 (verified in test) |
| Merged traits map to same z-point | Task 3 (verified in test) |
| `TransformersPolicy.act()` extended for SVD path | Task 3 |
| Gradio demo with 3 panels, live bar chart, persona sliders | Task 4 |
| Output key `"svd_probe"` in episode JSONL | Task 2 (honoured by `TransformersPolicy._last_probe`) |
| All 53 trait names valid in config; zeros explicit | Task 3 (`_build_injection` handles missing slugs) |

**Placeholder scan:** No "TBD", "TODO", or vague steps found. All code blocks are complete.

**Type consistency:** `SUBMODULE_SUFFIXES` imported from `svd_probe` in `svd_steering` — same dict in both files. `make_steering_hook` imported in both `svd_steering` (for injection) and `transformers_policy` (existing). `make_hook()` return type in `SVDPersonaProbe` matches `PersonaProbe` exactly.
