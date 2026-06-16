"""Unit tests for the multi-stream (resid + MLP) hook capture in
collect_activations.py. Uses a tiny hand-built model so no GPU, network,
or real Qwen weights are needed."""
import pytest
import torch
import torch.nn as nn

from scripts.collect_activations import (
    _extract_all_tokens, _resolve_streams, _resolve_submodule,
)


class _TinyMLP(nn.Module):
    def forward(self, x):
        return x * 2  # arbitrary deterministic transform


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _TinyMLP()

    def forward(self, x):
        return x + self.mlp(x)  # mimics a decoder block: resid = x + mlp(x)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _TinyLayer()

    def forward(self, input_ids):
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)  # [B, T, 4]
        return self.layer(x)


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "templated"

    def __call__(self, text, return_tensors=None):
        return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})


def test_resolve_streams_resid_and_mlp():
    model = _TinyModel()
    modules = _resolve_streams(model, "layer", ["resid", "mlp"])
    assert modules["resid"] is model.layer
    assert modules["mlp"] is model.layer.mlp


def test_resolve_streams_rejects_unknown_stream():
    model = _TinyModel()
    with pytest.raises(ValueError):
        _resolve_streams(model, "layer", ["bogus"])


def test_extract_all_tokens_captures_resid_and_mlp_in_one_pass():
    model = _TinyModel()
    tok = _FakeTokenizer()
    layer_modules = {
        "resid": _resolve_submodule(model, "layer"),
        "mlp": _resolve_submodule(model, "layer.mlp"),
    }
    captured = _extract_all_tokens(model, tok, "sys", "user", layer_modules, "cpu")
    assert set(captured) == {"resid", "mlp"}
    # mlp(x) = 2x, resid = x + mlp(x) = 3x -> resid == 1.5 * mlp, both from one pass
    assert torch.allclose(captured["resid"], captured["mlp"] * 1.5)


def test_extract_all_tokens_passes_enable_thinking_false():
    model = _TinyModel()
    tok = _FakeTokenizer()
    calls = []
    original = tok.apply_chat_template
    tok.apply_chat_template = lambda messages, **kw: (calls.append(kw), original(messages, **kw))[1]
    layer_modules = {"resid": _resolve_submodule(model, "layer")}
    _extract_all_tokens(model, tok, "sys", "user", layer_modules, "cpu")
    assert calls[-1]["enable_thinking"] is False
