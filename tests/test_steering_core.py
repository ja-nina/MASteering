import json, pathlib, tempfile
import torch
import pytest
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
