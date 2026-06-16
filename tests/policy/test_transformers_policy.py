import pytest
import torch
import torch.nn as nn

from testbed.policy.transformers_policy import _resolve_submodule, _SteeringSession
from testbed.steering.activation import make_steering_hook


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Identity()


def test_resolve_submodule_by_dotted_name():
    m = TinyModel()
    assert _resolve_submodule(m, "block") is m.block


def test_steering_session_adds_and_removes_hook():
    m = TinyModel()
    vec = torch.ones(3)
    hook = make_steering_hook(vec, coefficient=2.0)
    assert len(m.block._forward_hooks) == 0
    with _SteeringSession(m, "block", hook):
        assert len(m.block._forward_hooks) == 1
        out = m.block(torch.zeros((1, 1, 3)))
        assert torch.allclose(out, torch.full((1, 1, 3), 2.0))
    assert len(m.block._forward_hooks) == 0  # removed on exit


class _FakeBatch(dict):
    def to(self, device):
        return self


class _RecordingTokenizer:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        return "templated-text"

    def __call__(self, text, return_tensors=None):
        return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})


def test_build_inputs_disables_thinking_mode():
    from testbed.policy.transformers_policy import TransformersPolicy
    policy = TransformersPolicy.__new__(TransformersPolicy)
    policy.tokenizer = _RecordingTokenizer()
    policy.device = "cpu"
    policy._build_inputs("system prompt", "user prompt")
    assert policy.tokenizer.template_calls[-1]["enable_thinking"] is False


@pytest.mark.gpu
def test_transformers_policy_generates_with_qwen():
    from testbed.policy.transformers_policy import TransformersPolicy
    p = TransformersPolicy(model_id="Qwen/Qwen2.5-3B-Instruct")
    out = p.act("You are a helpful assistant.", "Say the word 'ok'.", "player_0", None)
    assert isinstance(out, str) and len(out) > 0
