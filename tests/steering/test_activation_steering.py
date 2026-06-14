import numpy as np
import torch
import torch.nn as nn

from testbed.steering.activation import ActivationSteering, make_steering_hook


def test_vector_loaded_from_npy(tmp_path):
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    p = tmp_path / "vec.npy"
    np.save(p, v)
    s = ActivationSteering(per_agent={
        "player_0": {"layer": "l", "vector_path": str(p), "coefficient": 2.0},
    })
    spec = s.steering_spec("player_0")
    assert spec is not None and spec.layer == "l" and spec.coefficient == 2.0
    vec = s.load_vector("player_0")
    assert torch.allclose(vec, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_hook_adds_scaled_vector_to_output():
    hidden = 4
    vec = torch.ones(hidden)
    hook = make_steering_hook(vec, coefficient=3.0)

    layer = nn.Identity()
    handle = layer.register_forward_hook(hook)
    x = torch.zeros((1, 2, hidden))  # (batch, seq, hidden)
    out = layer(x)
    handle.remove()
    # every position should have +3.0 added
    assert torch.allclose(out, torch.full((1, 2, hidden), 3.0))


def test_hook_handles_tuple_output():
    hidden = 3
    vec = torch.tensor([1.0, 0.0, 0.0])
    hook = make_steering_hook(vec, coefficient=5.0)

    class TupleLayer(nn.Module):
        def forward(self, x):
            return (x, "aux")  # transformer blocks often return tuples

    layer = TupleLayer()
    handle = layer.register_forward_hook(hook)
    x = torch.zeros((1, 1, hidden))
    out = layer(x)
    handle.remove()
    assert isinstance(out, tuple)
    assert torch.allclose(out[0], torch.tensor([[[5.0, 0.0, 0.0]]]))
    assert out[1] == "aux"


def test_unconfigured_agent_has_no_spec():
    s = ActivationSteering(per_agent={})
    assert s.steering_spec("player_3") is None
