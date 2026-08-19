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
