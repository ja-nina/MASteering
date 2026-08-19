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
