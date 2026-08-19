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
