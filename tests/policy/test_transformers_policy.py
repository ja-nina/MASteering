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


@pytest.mark.gpu
def test_transformers_policy_generates_with_qwen():
    from testbed.policy.transformers_policy import TransformersPolicy
    p = TransformersPolicy(model_id="Qwen/Qwen2.5-3B-Instruct")
    out = p.act("You are a helpful assistant.", "Say the word 'ok'.", "player_0", None)
    assert isinstance(out, str) and len(out) > 0
