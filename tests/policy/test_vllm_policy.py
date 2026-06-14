import pytest
from testbed.policy.vllm_policy import VLLMPolicy
from testbed.types import SteeringSpec


class _FakeCompletions:
    def create(self, **kwargs):
        class R:
            choices = [type("C", (), {"message": type("M", (), {"content": "hello"})()})()]
        # capture for assertions
        _FakeCompletions.last_kwargs = kwargs
        return R()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


def test_vllm_policy_returns_content():
    p = VLLMPolicy(model_id="Qwen/Qwen2.5-3B-Instruct", _client=_FakeClient())
    out = p.act("SYS", "USER", "player_0", None)
    assert out == "hello"
    msgs = _FakeCompletions.last_kwargs["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "SYS"
    assert msgs[1]["content"] == "USER"


def test_vllm_policy_rejects_activation_steering():
    p = VLLMPolicy(model_id="m", _client=_FakeClient())
    spec = SteeringSpec(method="activation", layer="l", vector_path="v.pt", coefficient=1.0)
    with pytest.raises(ValueError):
        p.act("SYS", "USER", "player_0", spec)
