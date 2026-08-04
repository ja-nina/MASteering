import json, pathlib, tempfile
import torch
import pytest
import matplotlib
matplotlib.use("Agg")
from notebooks.steering_core import ScheduleEntry, load_vectors, best_layer

def _make_fake_vectors(tmp_path):
    """Write two fake .pt vector files + one companion .json."""
    vecs = {10: torch.ones(8), 20: torch.ones(8) * 2, 29: torch.ones(8) * 3}
    torch.save(vecs, tmp_path / "sycophantic.pt")
    torch.save(vecs, tmp_path / "angry.pt")
    meta = {
        "sweep_scores": {
            "10": {"0.0": 0.1, "1.25": 0.4},   # delta = 0.3
            "20": {"0.0": 0.1, "1.25": 0.9},   # delta = 0.8  ← best
            "29": {"0.0": 0.1, "1.25": 0.5},   # delta = 0.4
        }
    }
    (tmp_path / "sycophantic.json").write_text(json.dumps(meta))

def test_schedule_entry_defaults():
    e = ScheduleEntry("sycophantic", layer=29, start=0, end=None, coeff=1.25)
    assert e.mode == "additive"
    assert e.end is None

def test_schedule_entry_explicit():
    e = ScheduleEntry("angry", layer=20, start=10, end=50, coeff=0.8, mode="adaptive")
    assert e.end == 50

def test_load_vectors_keys(tmp_path):
    _make_fake_vectors(tmp_path)
    vecs = load_vectors(str(tmp_path))
    assert set(vecs.keys()) == {"sycophantic", "angry"}

def test_load_vectors_layers(tmp_path):
    _make_fake_vectors(tmp_path)
    vecs = load_vectors(str(tmp_path))
    assert set(vecs["sycophantic"].keys()) == {10, 20, 29}
    assert vecs["sycophantic"][29].shape == (8,)

def test_best_layer_returns_correct(tmp_path):
    _make_fake_vectors(tmp_path)
    assert best_layer("sycophantic", str(tmp_path)) == 20

def test_best_layer_missing_json(tmp_path):
    _make_fake_vectors(tmp_path)
    # angry has no .json companion
    assert best_layer("angry", str(tmp_path)) is None

def test_load_model_returns_tuple(monkeypatch):
    """Stub out the heavy HF calls so this test runs without a GPU."""
    import types

    fake_tok = object()
    fake_model = types.SimpleNamespace(eval=lambda: None)

    import transformers
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: fake_tok
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: fake_model,
    )

    from notebooks.steering_core import load_model
    model, tok = load_model("fake/model", bits=None)
    assert tok is fake_tok


# ---------------------------------------------------------------------------
# Task 3 tests — hook machinery
# ---------------------------------------------------------------------------
import torch.nn as nn


class _FakeDecoder(nn.Module):
    """Two-layer fake transformer decoder for hook testing."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(30)])
        self.embed = nn.Embedding(100, 8)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return (h,)   # mimic HF output tuple


def test_token_counter_increments():
    """Token counter should equal number of generate() calls made."""
    from notebooks.steering_core import _build_hooks

    counter = {"t": 0}
    schedule = [ScheduleEntry("sycophantic", layer=10, start=0, end=None, coeff=1.0)]
    vectors = {"sycophantic": {10: torch.zeros(8)}}
    probe_traits = ["sycophantic"]
    probe_layers = [10]

    hooks, probe_store = _build_hooks(schedule, vectors, probe_traits, probe_layers, counter)
    # Simulate 3 forward passes through layer 10
    fake_module = nn.Linear(8, 8, bias=False)
    fake_output = (torch.zeros(1, 1, 8),)
    hook_fn = hooks[0][1]
    for _ in range(3):
        hook_fn(fake_module, None, fake_output)
    assert counter["t"] == 3


def test_additive_steering_applied():
    """Additive hook should add coeff*vec to hidden state when in window."""
    from notebooks.steering_core import _build_hooks

    counter = {"t": 0}
    vec = torch.ones(8)
    schedule = [ScheduleEntry("sycophantic", layer=29, start=0, end=None, coeff=2.0)]
    vectors = {"sycophantic": {29: vec}}
    probe_traits = []
    probe_layers = []

    hooks, _ = _build_hooks(schedule, vectors, probe_traits, probe_layers, counter)
    fake_module = nn.Linear(8, 8, bias=False)
    hidden = torch.zeros(1, 1, 8)
    result = hooks[0][1](fake_module, None, (hidden,))
    # should add 2.0 * ones(8)
    assert torch.allclose(result[0], hidden + 2.0 * vec)


def test_steering_not_applied_outside_window():
    """Hook should not steer when t is outside [start, end)."""
    from notebooks.steering_core import _build_hooks

    counter = {"t": 5}  # already past the window
    vec = torch.ones(8)
    schedule = [ScheduleEntry("sycophantic", layer=29, start=0, end=3, coeff=2.0)]
    vectors = {"sycophantic": {29: vec}}
    probe_traits = []
    probe_layers = []

    hooks, _ = _build_hooks(schedule, vectors, probe_traits, probe_layers, counter)
    fake_module = nn.Linear(8, 8, bias=False)
    hidden = torch.zeros(1, 1, 8)
    result = hooks[0][1](fake_module, None, (hidden,))
    assert torch.allclose(result[0], hidden)  # unchanged


def test_probe_records_scores():
    """Probe hook should append one score dict per token per layer."""
    from notebooks.steering_core import _build_hooks

    counter = {"t": 0}
    vec = torch.ones(8) / (8 ** 0.5)  # unit vector
    schedule = []
    vectors = {"sycophantic": {10: vec * 4.0}}  # will be used by probe too
    probe_traits = ["sycophantic"]
    probe_layers = [10]

    hooks, probe_store = _build_hooks(schedule, vectors, probe_traits, probe_layers, counter)
    fake_module = nn.Linear(8, 8, bias=False)
    hidden = torch.ones(1, 1, 8)
    hooks[0][1](fake_module, None, (hidden,))

    assert 10 in probe_store
    assert len(probe_store[10]) == 1
    assert "sycophantic" in probe_store[10][0]


# ---------------------------------------------------------------------------
# Task 4 tests — plot_probe
# ---------------------------------------------------------------------------

def test_plot_probe_returns_figure():
    from notebooks.steering_core import plot_probe
    token_strings = ["Hello", " world", "!"]
    probe_data = {
        29: [
            {"sycophantic": 0.1, "angry": -0.2},
            {"sycophantic": 0.3, "angry": -0.1},
            {"sycophantic": 0.5, "angry": 0.0},
        ]
    }
    schedule = [ScheduleEntry("sycophantic", layer=29, start=0, end=None, coeff=1.25)]
    import matplotlib.figure
    fig = plot_probe(token_strings, probe_data, schedule, layer=29, traits=["sycophantic", "angry"])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_probe_shades_none_end():
    """A schedule entry with end=None should produce a shaded region to the last token."""
    from notebooks.steering_core import plot_probe
    token_strings = ["a", "b", "c", "d"]
    probe_data = {10: [{"sycophantic": float(i) * 0.1} for i in range(4)]}
    schedule = [ScheduleEntry("sycophantic", layer=10, start=1, end=None, coeff=1.0)]
    fig = plot_probe(token_strings, probe_data, schedule, layer=10, traits=["sycophantic"])
    # Just check it doesn't raise and returns a figure
    assert fig is not None


# ---------------------------------------------------------------------------
# Task 6 tests — steering_demo
# ---------------------------------------------------------------------------

def test_parse_schedule_from_rows():
    """_parse_schedule converts Dataframe rows into ScheduleEntry list."""
    from notebooks.steering_demo import _parse_schedule

    rows = [
        ["sycophantic", "29", "0", "",   "1.25", "additive"],
        ["angry",       "20", "10", "50", "0.8",  "adaptive"],
    ]
    entries = _parse_schedule(rows)
    assert len(entries) == 2
    assert entries[0].end is None        # blank end → None
    assert entries[1].end == 50
    assert entries[0].layer == 29
    assert entries[1].mode == "adaptive"


def test_parse_schedule_skips_short_rows():
    from notebooks.steering_demo import _parse_schedule
    rows = [
        ["sycophantic", "29", "0", "", "1.25", "additive"],  # valid
        ["angry"],  # too short, should be skipped
        [],  # empty, should be skipped
    ]
    entries = _parse_schedule(rows)
    assert len(entries) == 1
    assert entries[0].trait == "sycophantic"
