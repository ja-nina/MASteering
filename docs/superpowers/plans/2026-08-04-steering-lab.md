# Steering Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a token-level activation steering lab consisting of a shared core module, a researcher notebook, and a shareable Gradio demo — all running Qwen3-14B in 4-bit on a Linux/WSL2 server.

**Architecture:** A single `notebooks/steering_core.py` holds all reusable logic (model loading, vector loading, hook construction, generation, plotting). `steering_lab.ipynb` and `steering_demo.py` are thin callers — no logic is duplicated between them. Hooks are registered as standard PyTorch forward hooks inside a context manager so they are always cleaned up.

**Tech Stack:** PyTorch, HuggingFace Transformers, bitsandbytes (4-bit), Gradio ≥4.0, matplotlib, nbformat (notebook creation).

## Global Constraints

- `bitsandbytes>=0.43` required for 4-bit; falls back gracefully to bf16 when `bits=None`
- `gradio>=4.0` required for `gr.Dataframe` column type support
- All paths use `os.path.expandvars` so `${PERSONA_VECTORS_ROOT}` works
- Vector files follow PersVecGen format: `.pt` files are dicts `{layer_int: tensor}` or plain tensors
- `ScheduleEntry.end = None` means "steer until the last generated token"
- Token counter uses the **highest** scheduled/probe layer as the tick layer — so all hooks in one forward pass see the same `t` before it increments
- Additive and adaptive formulas are copied verbatim from `testbed/steering/activation.py`
- Model launched with `model.eval()` and `torch.no_grad()` during generation

---

## File Map

| File | Created/Modified | Responsibility |
|------|-----------------|----------------|
| `notebooks/steering_core.py` | Create | `ScheduleEntry`, `load_model`, `load_vectors`, `best_layer`, `run_generation`, `plot_probe` |
| `notebooks/steering_lab.ipynb` | Create | 5-cell researcher notebook |
| `notebooks/steering_demo.py` | Create | Gradio UI app |
| `tests/test_steering_core.py` | Create | Unit tests for core logic |

---

## Task 1: `ScheduleEntry` + `load_vectors` + `best_layer`

**Files:**
- Create: `notebooks/steering_core.py`
- Create: `tests/test_steering_core.py`

**Interfaces:**
- Produces:
  - `ScheduleEntry(trait, layer, start, end, coeff, mode="additive")` dataclass
  - `load_vectors(vectors_dir: str) -> Dict[str, Dict[int, torch.Tensor]]`
  - `best_layer(trait: str, vectors_dir: str) -> Optional[int]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_steering_core.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:/Users/ismyn/UNI/MPI/Thesis/MA_Environments
python -m pytest tests/test_steering_core.py -v
```
Expected: `ModuleNotFoundError: No module named 'notebooks.steering_core'`

- [ ] **Step 3: Implement**

```python
# notebooks/steering_core.py
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScheduleEntry:
    trait: str
    layer: int
    start: int
    end: Optional[int]   # None = steer until end of generation
    coeff: float
    mode: str = "additive"  # "additive" | "adaptive"


# ---------------------------------------------------------------------------
# Vector loading
# ---------------------------------------------------------------------------

def load_vectors(vectors_dir: str) -> Dict[str, Dict[int, torch.Tensor]]:
    """Load all *.pt trait vectors from a PersVecGen bf16 directory.

    Returns {trait_name: {layer_int: tensor}}.
    Plain-tensor files (not dicts) are stored as {-1: tensor}.
    """
    result: Dict[str, Dict[int, torch.Tensor]] = {}
    for pt_path in sorted(pathlib.Path(vectors_dir).glob("*.pt")):
        trait = pt_path.stem
        loaded = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            result[trait] = {int(k): v.float() for k, v in loaded.items()}
        else:
            result[trait] = {-1: loaded.float()}
    return result


def best_layer(trait: str, vectors_dir: str) -> Optional[int]:
    """Return the integer layer with the highest sweep delta for a trait.

    Reads the companion <trait>.json file (PersVecGen metadata).
    Returns None if the file does not exist.
    """
    json_path = pathlib.Path(vectors_dir) / f"{trait}.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        meta = json.load(f)
    sweep = meta.get("sweep_scores", {})
    if not sweep:
        return None

    def _delta(lk: str) -> float:
        scores = {float(k): float(v) for k, v in sweep[lk].items()}
        baseline = scores.get(0.0, min(scores.values()))
        return max(scores.values()) - baseline

    return int(max(sweep.keys(), key=_delta))
```

- [ ] **Step 4: Add `notebooks/__init__.py` so the package is importable**

```bash
touch notebooks/__init__.py
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_steering_core.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add notebooks/__init__.py notebooks/steering_core.py tests/test_steering_core.py
git commit -m "feat: steering_core scaffold — ScheduleEntry, load_vectors, best_layer"
```

---

## Task 2: `load_model`

**Files:**
- Modify: `notebooks/steering_core.py`
- Modify: `tests/test_steering_core.py`

**Interfaces:**
- Consumes: nothing from prior tasks
- Produces: `load_model(model_id: str = "Qwen/Qwen3-14B", bits: Optional[int] = 4) -> Tuple[model, tokenizer]`

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_steering_core.py

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_steering_core.py::test_load_model_returns_tuple -v
```
Expected: FAIL — `load_model` not defined.

- [ ] **Step 3: Implement**

Append to `notebooks/steering_core.py`:

```python
# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_id: str = "Qwen/Qwen3-14B",
    bits: Optional[int] = 4,
) -> Tuple[object, object]:
    """Load model and tokenizer.

    bits=4  — BitsAndBytesConfig 4-bit NF4 (requires bitsandbytes + CUDA)
    bits=None — bf16, no quantization
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if bits == 4:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_steering_core.py::test_load_model_returns_tuple -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/steering_core.py tests/test_steering_core.py
git commit -m "feat: steering_core — load_model with 4-bit / bf16"
```

---

## Task 3: `run_generation` — hook machinery

**Files:**
- Modify: `notebooks/steering_core.py`
- Modify: `tests/test_steering_core.py`

**Interfaces:**
- Consumes: `ScheduleEntry`, `load_vectors` output
- Produces:
  ```python
  run_generation(
      model, tokenizer, prompt: str,
      schedule: List[ScheduleEntry],
      vectors: Dict[str, Dict[int, torch.Tensor]],
      probe_traits: List[str],
      probe_layers: List[int],
      max_new_tokens: int = 256,
      enable_thinking: bool = False,
  ) -> Tuple[str, List[str], Dict[int, List[Dict[str, float]]]]
  # returns: (text, token_strings, probe_data)
  # probe_data = {layer_int: [{trait: score, ...}, ...]}  one dict per token
  ```

**Token counter design:** `token_counter = {"t": 0}`. The hook registered at the **highest** layer among `(schedule layers ∪ probe_layers)` increments the counter **after** doing its work, so all hooks in a single forward pass read the same `t`.

- [ ] **Step 1: Write failing tests using a tiny fake model**

```python
# append to tests/test_steering_core.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_steering_core.py::test_token_counter_increments tests/test_steering_core.py::test_additive_steering_applied tests/test_steering_core.py::test_steering_not_applied_outside_window tests/test_steering_core.py::test_probe_records_scores -v
```
Expected: all FAIL — `_build_hooks` not defined.

- [ ] **Step 3: Implement `_build_hooks` and `run_generation`**

Append to `notebooks/steering_core.py`:

```python
# ---------------------------------------------------------------------------
# Hook construction
# ---------------------------------------------------------------------------

def _resolve_submodule(model, dotted_name: str):
    obj = model
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


def _build_hooks(
    schedule: List[ScheduleEntry],
    vectors: Dict[str, Dict[int, torch.Tensor]],
    probe_traits: List[str],
    probe_layers: List[int],
    token_counter: Dict[str, int],
) -> Tuple[List[Tuple[str, callable]], Dict[int, List[Dict[str, float]]]]:
    """Build all forward hooks and return (hook_list, probe_store).

    hook_list  — list of (layer_dotpath, hook_fn) pairs ready for registration
    probe_store — {layer_int: []} — caller reads this after generation
    """
    # Determine tick layer: highest int across schedule + probe layers
    all_layers = set(e.layer for e in schedule) | set(probe_layers)
    tick_layer = max(all_layers) if all_layers else None

    # Group schedule entries by layer
    by_layer: Dict[int, List[ScheduleEntry]] = {}
    for entry in schedule:
        by_layer.setdefault(entry.layer, []).append(entry)

    # Pre-load probe vectors: {trait: {layer: (v_hat, norm)}}
    probe_vecs: Dict[str, Dict[int, Tuple[torch.Tensor, float]]] = {}
    for trait in probe_traits:
        probe_vecs[trait] = {}
        for pl in probe_layers:
            if trait in vectors and pl in vectors[trait]:
                v = vectors[trait][pl].float()
                norm = v.norm().item()
                probe_vecs[trait][pl] = (v / norm if norm > 0 else v, norm)

    probe_store: Dict[int, List[Dict[str, float]]] = {pl: [] for pl in probe_layers}

    hook_list: List[Tuple[str, callable]] = []

    # Collect all layers that need a hook (steering layers ∪ probe layers)
    all_hook_layers = set(by_layer.keys()) | set(probe_layers)

    for layer_int in sorted(all_hook_layers):
        layer_path = f"model.layers.{layer_int}"
        entries = by_layer.get(layer_int, [])
        is_tick = (layer_int == tick_layer)
        pl = layer_int if layer_int in probe_layers else None

        def _make_hook(entries=entries, pl=pl, is_tick=is_tick, layer_int=layer_int):
            def hook(module, inputs, output):
                t = token_counter["t"]
                hidden = output[0] if isinstance(output, tuple) else output

                # ── steering ────────────────────────────────────────────────
                active = [
                    e for e in entries
                    if e.start <= t and (e.end is None or t < e.end)
                ]
                if active:
                    additive_sum = None
                    for e in active:
                        if e.layer not in vectors.get(e.trait, {}):
                            continue
                        v = vectors[e.trait][e.layer].float().to(hidden.device).to(hidden.dtype)
                        if e.mode == "additive":
                            delta = e.coeff * v
                            additive_sum = delta if additive_sum is None else additive_sum + delta
                        else:  # adaptive
                            norm = v.norm()
                            if norm == 0:
                                continue
                            v_hat = v / norm
                            target = e.coeff * norm
                            proj = (hidden * v_hat).sum(dim=-1, keepdim=True)
                            correction = target - proj
                            correction = (
                                correction.clamp(min=0) if e.coeff >= 0
                                else correction.clamp(max=0)
                            )
                            hidden = hidden + correction * v_hat

                    if additive_sum is not None:
                        hidden = hidden + additive_sum.to(hidden.device).to(hidden.dtype)

                # ── probing ──────────────────────────────────────────────────
                if pl is not None:
                    scores: Dict[str, float] = {}
                    for trait, layer_vecs in probe_vecs.items():
                        if pl in layer_vecs:
                            v_hat, norm = layer_vecs[pl]
                            v_hat = v_hat.to(hidden.device).to(hidden.dtype)
                            # mean over batch & sequence dims, then dot
                            h_mean = hidden.float().mean(dim=1).squeeze(0)
                            score = float((h_mean * v_hat.float()).sum()) / norm
                            scores[trait] = score
                    probe_store[pl].append(scores)

                # ── tick ─────────────────────────────────────────────────────
                if is_tick:
                    token_counter["t"] += 1

                if isinstance(output, tuple):
                    return (hidden,) + tuple(output[1:])
                return hidden

            return hook

        hook_list.append((layer_path, _make_hook()))

    return hook_list, probe_store


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation(
    model,
    tokenizer,
    prompt: str,
    schedule: List[ScheduleEntry],
    vectors: Dict[str, Dict[int, torch.Tensor]],
    probe_traits: List[str],
    probe_layers: List[int],
    max_new_tokens: int = 256,
    enable_thinking: bool = False,
) -> Tuple[str, List[str], Dict[int, List[Dict[str, float]]]]:
    """Run steered generation and return (text, token_strings, probe_data)."""
    messages = [{"role": "user", "content": prompt}]
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if enable_thinking:
        text_input += "<think>\n"

    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    token_counter = {"t": 0}
    hook_list, probe_store = _build_hooks(
        schedule, vectors, probe_traits, probe_layers, token_counter
    )

    # Register hooks
    handles = []
    for layer_path, hook_fn in hook_list:
        module = _resolve_submodule(model, layer_path)
        handles.append(module.register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        for h in handles:
            h.remove()

    gen_ids = out[0][input_len:]

    if enable_thinking:
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        eos = tokenizer.eos_token or ""
        text = text.rstrip().removesuffix(eos).rstrip()
    else:
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    token_strings = [tokenizer.decode([tid]) for tid in gen_ids.tolist()]

    return text, token_strings, probe_store
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_steering_core.py -v
```
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/steering_core.py tests/test_steering_core.py
git commit -m "feat: steering_core — _build_hooks and run_generation"
```

---

## Task 4: `plot_probe`

**Files:**
- Modify: `notebooks/steering_core.py`
- Modify: `tests/test_steering_core.py`

**Interfaces:**
- Consumes: `run_generation` outputs
- Produces:
  ```python
  plot_probe(
      token_strings: List[str],
      probe_data: Dict[int, List[Dict[str, float]]],
      schedule: List[ScheduleEntry],
      layer: int,
      traits: List[str],
  ) -> matplotlib.figure.Figure
  ```

- [ ] **Step 1: Write failing tests**

```python
# append to tests/test_steering_core.py
import matplotlib
matplotlib.use("Agg")

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_steering_core.py::test_plot_probe_returns_figure tests/test_steering_core.py::test_plot_probe_shades_none_end -v
```
Expected: FAIL — `plot_probe` not defined.

- [ ] **Step 3: Implement**

Append to `notebooks/steering_core.py`:

```python
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Categorical palette — adjacent-pairs CVD-safe
_CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300"]


def plot_probe(
    token_strings: List[str],
    probe_data: Dict[int, List[Dict[str, float]]],
    schedule: List[ScheduleEntry],
    layer: int,
    traits: List[str],
) -> "matplotlib.figure.Figure":
    """Return a matplotlib Figure showing per-token probe scores for one layer.

    - One line per trait in `traits`
    - Translucent shaded bands for each ScheduleEntry window (end=None → last token)
    - Horizontal dashed line at y=0
    - X-axis labeled with token strings every 10 tokens
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    per_layer = probe_data.get(layer, [])
    n = len(per_layer)
    xs = list(range(n))

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#f5f4f0")
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Trait lines
    for ci, trait in enumerate(traits):
        ys = [step.get(trait, float("nan")) for step in per_layer]
        ax.plot(xs, ys, color=_CAT[ci % len(_CAT)], lw=1.8, label=trait)

    # Shaded bands for each schedule entry
    band_colors = _CAT[len(traits) % len(_CAT):]  # offset to avoid clashing with trait lines
    for bi, entry in enumerate(schedule):
        x0 = entry.start
        x1 = (entry.end - 1) if entry.end is not None else (n - 1)
        x1 = min(x1, n - 1)
        color = band_colors[bi % len(band_colors)]
        ax.axvspan(x0, x1, alpha=0.12, color=color,
                   label=f"{entry.trait} α={entry.coeff}")

    ax.axhline(0, color="#c8c6be", lw=0.8, ls="--")

    # X-axis: show token string every 10 ticks
    tick_step = max(1, n // 20)
    tick_positions = list(range(0, n, tick_step))
    tick_labels = [token_strings[i] if i < len(token_strings) else "" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)

    ax.set_xlabel("Generated token index", fontsize=9)
    ax.set_ylabel("Projection / ||v||", fontsize=9)
    ax.set_title(f"Persona probe — layer {layer}", fontsize=11, fontweight="semibold")
    ax.legend(loc="upper right", fontsize=7.5, ncol=3, framealpha=0.85, frameon=True)
    ax.grid(axis="y", alpha=0.25, lw=0.6)

    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_steering_core.py -v
```
Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add notebooks/steering_core.py tests/test_steering_core.py
git commit -m "feat: steering_core — plot_probe with shaded schedule bands"
```

---

## Task 5: `steering_lab.ipynb`

**Files:**
- Create: `notebooks/steering_lab.ipynb`

**Interfaces:**
- Consumes: all public functions from `steering_core.py`
- Produces: a 5-cell notebook that runs top-to-bottom without errors (when model is available)

No unit tests for the notebook itself; verify by running all cells.

- [ ] **Step 1: Create the notebook with nbformat**

Run this Python script once to generate the file:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# ── Cell 1: Setup ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
import sys, os
sys.path.insert(0, os.path.abspath(".."))   # make steering_core importable

from notebooks.steering_core import (
    ScheduleEntry, load_model, load_vectors, best_layer,
    run_generation, plot_probe,
)
import matplotlib.pyplot as plt

VECTORS_DIR = os.path.expandvars(
    "${PERSONA_VECTORS_ROOT}/bf16"
)
MODEL_ID = "Qwen/Qwen3-14B"

# Load once — takes ~2 min on first run
model, tokenizer = load_model(MODEL_ID, bits=4)
vectors = load_vectors(VECTORS_DIR)
print(f"Loaded {len(vectors)} trait vectors.")
"""))

# ── Cell 2: Inspect ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
print("Available traits:")
for trait in sorted(vectors.keys()):
    bl = best_layer(trait, VECTORS_DIR)
    print(f"  {trait:<35} best layer: {bl}")
"""))

# ── Cell 3: Schedule (edit this cell each experiment) ─────────────────────
cells.append(nbf.v4.new_code_cell("""\
prompt = "You are a player in the Mafia game. Your role is Mafia. Describe your strategy."

schedule = [
    ScheduleEntry("sycophantic", layer=29, start=0,  end=None, coeff=1.25, mode="additive"),
    ScheduleEntry("angry",       layer=29, start=30, end=None, coeff=0.8,  mode="additive"),
]

probe_traits = ["sycophantic", "angry", "ethical", "trustworthiness"]
probe_layers = [10, 20, 29]
enable_thinking = False
max_new_tokens = 256
"""))

# ── Cell 4: Generate ───────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
text, token_strings, probe_data = run_generation(
    model, tokenizer, prompt,
    schedule=schedule,
    vectors=vectors,
    probe_traits=probe_traits,
    probe_layers=probe_layers,
    max_new_tokens=max_new_tokens,
    enable_thinking=enable_thinking,
)
print(f"Generated {len(token_strings)} tokens.\\n")
print(text)
"""))

# ── Cell 5: Plot ───────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
for layer in probe_layers:
    fig = plot_probe(token_strings, probe_data, schedule, layer=layer, traits=probe_traits)
    plt.show()
"""))

nb.cells = cells
with open("notebooks/steering_lab.ipynb", "w") as f:
    nbf.write(nb, f)
print("Written: notebooks/steering_lab.ipynb")
```

Save this as `scripts/create_steering_lab_notebook.py` and run:

```bash
python scripts/create_steering_lab_notebook.py
```

- [ ] **Step 2: Verify the file is valid JSON**

```bash
python -c "import json; json.load(open('notebooks/steering_lab.ipynb')); print('valid')"
```
Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add notebooks/steering_lab.ipynb scripts/create_steering_lab_notebook.py
git commit -m "feat: steering_lab.ipynb — 5-cell researcher notebook"
```

---

## Task 6: `steering_demo.py` — Gradio app

**Files:**
- Create: `notebooks/steering_demo.py`
- Modify: `tests/test_steering_core.py`

**Interfaces:**
- Consumes: all public functions from `steering_core.py`
- Produces: a runnable Gradio app; `_parse_schedule(df_rows, vectors)` helper tested in isolation

- [ ] **Step 1: Write failing test for the schedule parser**

```python
# append to tests/test_steering_core.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_steering_core.py::test_parse_schedule_from_rows -v
```
Expected: FAIL — `notebooks.steering_demo` not found.

- [ ] **Step 3: Implement `steering_demo.py`**

```python
# notebooks/steering_demo.py
"""Gradio demo for token-level activation steering.

Launch:
    python notebooks/steering_demo.py            # local only
    python notebooks/steering_demo.py --share    # public Gradio link
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from PIL import Image

from notebooks.steering_core import (
    ScheduleEntry,
    best_layer,
    load_model,
    load_vectors,
    plot_probe,
    run_generation,
)

# ---------------------------------------------------------------------------
# Schedule parsing helper (also tested directly)
# ---------------------------------------------------------------------------

def _parse_schedule(rows: List[List]) -> List[ScheduleEntry]:
    """Convert Gradio Dataframe rows into ScheduleEntry objects.

    Row format: [trait, layer, start, end, coeff, mode]
    Blank or None `end` cell → ScheduleEntry.end = None (steer to end).
    """
    entries = []
    for row in rows:
        if not row or not row[0]:
            continue
        trait = str(row[0]).strip()
        layer = int(row[1])
        start = int(row[2])
        end_raw = str(row[3]).strip() if row[3] not in (None, "") else ""
        end: Optional[int] = int(end_raw) if end_raw else None
        coeff = float(row[4])
        mode = str(row[5]).strip() if row[5] else "additive"
        entries.append(ScheduleEntry(trait, layer, start, end, coeff, mode))
    return entries


# ---------------------------------------------------------------------------
# Startup: load model + vectors once
# ---------------------------------------------------------------------------

VECTORS_DIR = os.path.expandvars("${PERSONA_VECTORS_ROOT}/bf16")
MODEL_ID = "Qwen/Qwen3-14B"

print("Loading model…")
model, tokenizer = load_model(MODEL_ID, bits=4)
print("Loading vectors…")
vectors = load_vectors(VECTORS_DIR)
ALL_TRAITS = sorted(vectors.keys())
print(f"Ready — {len(ALL_TRAITS)} traits available.")


# ---------------------------------------------------------------------------
# Generate callback
# ---------------------------------------------------------------------------

def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return Image.open(buf).copy()


def generate_callback(
    prompt: str,
    schedule_df,
    probe_traits_selected: List[str],
    probe_layer: int,
    max_new_tokens: int,
    enable_thinking: bool,
):
    rows = schedule_df.values.tolist() if hasattr(schedule_df, "values") else schedule_df
    schedule = _parse_schedule(rows)

    if not probe_traits_selected:
        probe_traits_selected = list({e.trait for e in schedule})

    text, token_strings, probe_data = run_generation(
        model, tokenizer, prompt,
        schedule=schedule,
        vectors=vectors,
        probe_traits=probe_traits_selected,
        probe_layers=[probe_layer],
        max_new_tokens=int(max_new_tokens),
        enable_thinking=bool(enable_thinking),
    )

    fig = plot_probe(
        token_strings, probe_data, schedule,
        layer=probe_layer,
        traits=probe_traits_selected,
    )
    return text, _fig_to_pil(fig)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_DEFAULT_SCHEDULE = [
    ["sycophantic", 29, 0, "", 1.25, "additive"],
    ["angry",       29, 30, "", 0.8, "additive"],
]

with gr.Blocks(title="Steering Lab") as demo:
    gr.Markdown("## Activation Steering Lab — Qwen3-14B 4-bit")
    with gr.Row():
        # ── Left column ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            prompt_box = gr.Textbox(
                label="Prompt",
                value="You are a player in the Mafia game. Your role is Mafia. Describe your strategy.",
                lines=4,
            )
            schedule_table = gr.Dataframe(
                headers=["trait", "layer", "start", "end (blank=∞)", "coeff", "mode"],
                datatype=["str", "number", "number", "str", "number", "str"],
                value=_DEFAULT_SCHEDULE,
                row_count=(4, "dynamic"),
                col_count=(6, "fixed"),
                label="Steering schedule",
            )
            probe_traits_box = gr.CheckboxGroup(
                choices=ALL_TRAITS,
                value=["sycophantic", "angry", "ethical", "trustworthiness"],
                label="Probe traits",
            )
            probe_layer_radio = gr.Radio(
                choices=[10, 20, 29],
                value=29,
                label="Probe layer",
            )
            max_tokens_slider = gr.Slider(
                minimum=64, maximum=512, step=32, value=256,
                label="Max new tokens",
            )
            thinking_checkbox = gr.Checkbox(
                value=False,
                label="Enable reasoning (<think> mode)",
            )
            generate_btn = gr.Button("Generate", variant="primary")

        # ── Right column ───────────────────────────────────────────────────
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Generated text", lines=12)
            output_plot = gr.Image(label="Probe scores", type="pil")

    generate_btn.click(
        fn=generate_callback,
        inputs=[
            prompt_box, schedule_table, probe_traits_box,
            probe_layer_radio, max_tokens_slider, thinking_checkbox,
        ],
        outputs=[output_text, output_plot],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.launch(share=args.share, server_port=args.port)
```

- [ ] **Step 4: Run the schedule parser test**

```bash
python -m pytest tests/test_steering_core.py::test_parse_schedule_from_rows -v
```
Expected: PASS.

- [ ] **Step 5: Smoke-test the import (no GPU needed)**

```bash
python -c "
import sys, os, unittest.mock as mock
# Stub heavy dependencies so the import is fast
with mock.patch('notebooks.steering_core.load_model', return_value=(None, None)), \
     mock.patch('notebooks.steering_core.load_vectors', return_value={}):
    # Can't import steering_demo directly because it runs load_model at module level.
    # Verify _parse_schedule works standalone instead.
    from notebooks.steering_demo import _parse_schedule
    rows = [['sycophantic', '29', '0', '', '1.25', 'additive']]
    result = _parse_schedule(rows)
    assert result[0].end is None
    print('_parse_schedule OK')
"
```
Expected: `_parse_schedule OK`

Note: full end-to-end smoke test requires the model to be loaded on the server. Run there with:

```bash
python notebooks/steering_demo.py --share
```

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/test_steering_core.py -v
```
Expected: all 13 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add notebooks/steering_demo.py tests/test_steering_core.py
git commit -m "feat: steering_demo.py — Gradio app with schedule builder and probe plot"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|-----------------|------|
| `ScheduleEntry` with `end=None` | Task 1 |
| `load_model` 4-bit / bf16 | Task 2 |
| `load_vectors` PersVecGen format | Task 1 |
| `best_layer` helper | Task 1 |
| Token-gated composite hooks | Task 3 |
| Overlapping multi-vector schedules | Task 3 |
| Additive + adaptive modes | Task 3 |
| Per-token probe scores | Task 3 |
| `plot_probe` with shaded bands | Task 4 |
| `end=None` bands extend to last token in plot | Task 4 |
| `enable_thinking` toggle | Task 3 |
| 5-cell notebook | Task 5 |
| Gradio two-column UI | Task 6 |
| `--share` flag | Task 6 |
| Schedule Dataframe with blank end → None | Task 6 |

All requirements covered. ✓

### Type consistency

- `ScheduleEntry` defined in Task 1, consumed identically in Tasks 3, 4, 6. ✓
- `probe_data` type `Dict[int, List[Dict[str, float]]]` produced by Task 3, consumed by Task 4 and Task 6. ✓
- `_parse_schedule` returns `List[ScheduleEntry]` — matches what `generate_callback` passes to `run_generation`. ✓

### Placeholder scan

No TBDs, TODOs, or vague steps found. ✓
