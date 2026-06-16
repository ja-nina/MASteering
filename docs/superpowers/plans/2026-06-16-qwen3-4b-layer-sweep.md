# Qwen3-4B Migration + Full-Layer SAE Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the steering testbed's base model to `Qwen/Qwen3-4B` (non-thinking mode, matched sampling), and extend the SAE activation pipeline to sweep every layer's residual stream and MLP output while keeping raw-activation disk usage bounded to roughly one layer at a time.

**Architecture:** `TransformersPolicy` gains `top_p`/`top_k` and a hardcoded `enable_thinking=False` chat-template flag. `collect_activations.py` is extended to hook multiple submodules (`resid`, `mlp`) per layer in a single forward pass and write one file set per stream. A new `scripts/run_layer_sweep.py` orchestrates `collect_activations.py` → `train_sae.py` → `find_tom_features.py` per layer/stream as subprocesses, deleting the large raw-activation file immediately after each combo finishes.

**Tech Stack:** Python, PyTorch, HuggingFace `transformers`, pytest. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-06-16-qwen3-4b-layer-sweep-design.md](../specs/2026-06-16-qwen3-4b-layer-sweep-design.md)

---

### Task 1: `enable_thinking=False` in `TransformersPolicy._build_inputs`

**Files:**
- Modify: `testbed/policy/transformers_policy.py:57-62`
- Test: `tests/policy/test_transformers_policy.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/policy/test_transformers_policy.py` (new imports go at the top alongside the existing `pytest`/`torch`/`nn` imports):

```python
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
```

`TransformersPolicy.__new__` bypasses `__init__` so no real model/tokenizer is loaded — this test needs no GPU and no network.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/policy/test_transformers_policy.py::test_build_inputs_disables_thinking_mode -v`
Expected: FAIL with `KeyError: 'enable_thinking'`

- [ ] **Step 3: Implement**

In `testbed/policy/transformers_policy.py`, change `_build_inputs` (currently lines 57-62):

```python
    def _build_inputs(self, system_prompt: str, user_prompt: str):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        return self.tokenizer(text, return_tensors="pt").to(self.device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/policy/test_transformers_policy.py::test_build_inputs_disables_thinking_mode -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add testbed/policy/transformers_policy.py tests/policy/test_transformers_policy.py
git commit -m "feat: disable Qwen3 thinking mode in TransformersPolicy chat template"
```

---

### Task 2: `top_p`/`top_k` sampling support in `TransformersPolicy`

**Files:**
- Modify: `testbed/policy/transformers_policy.py:36-53` (`__init__`), `:64-72` (`_generate`)
- Test: `tests/policy/test_transformers_policy.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/policy/test_transformers_policy.py`:

```python
class _RecordingModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        extra = torch.zeros((input_ids.shape[0], 1), dtype=torch.long)
        return torch.cat([input_ids, extra], dim=1)


class _DecodingTokenizer:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "ok"


def test_generate_passes_top_p_and_top_k_to_model():
    from testbed.policy.transformers_policy import TransformersPolicy
    policy = TransformersPolicy.__new__(TransformersPolicy)
    policy.model = _RecordingModel()
    policy.tokenizer = _DecodingTokenizer()
    policy.temperature = 0.7
    policy.top_p = 0.8
    policy.top_k = 20
    policy.max_new_tokens = 8
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    policy._generate(inputs)
    call_kwargs = policy.model.calls[0]
    assert call_kwargs["top_p"] == 0.8
    assert call_kwargs["top_k"] == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/policy/test_transformers_policy.py::test_generate_passes_top_p_and_top_k_to_model -v`
Expected: FAIL with `KeyError: 'top_p'`

- [ ] **Step 3: Implement**

In `testbed/policy/transformers_policy.py`, change `__init__` (currently lines 36-53):

```python
class TransformersPolicy:
    def __init__(self, model_id: str = "Qwen/Qwen3-4B",
                 temperature: float = 0.7, top_p: float = 0.8, top_k: int = 20,
                 max_new_tokens: int = 256,
                 device: Optional[str] = None,
                 steering: Optional[object] = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # GTX 16xx / Turing cards don't support bfloat16; use float16 on CUDA
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()
        # the steering method (ActivationSteering) used to load vectors by agent
        self.steering = steering
```

And `_generate` (currently lines 64-72):

```python
    def _generate(self, inputs) -> str:
        import torch
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5),
                top_p=self.top_p, top_k=self.top_k,
                pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/policy/test_transformers_policy.py::test_generate_passes_top_p_and_top_k_to_model -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add testbed/policy/transformers_policy.py tests/policy/test_transformers_policy.py
git commit -m "feat: add top_p/top_k sampling support to TransformersPolicy"
```

---

### Task 3: Wire `top_p`/`top_k` through `build_policy`

**Files:**
- Modify: `testbed/config.py:27-35`
- Test: `tests/test_config_and_e2e.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config_and_e2e.py`:

```python
def test_build_policy_passes_top_p_top_k_for_transformers_backend(monkeypatch):
    captured = {}

    class _StubPolicy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "testbed.policy.transformers_policy.TransformersPolicy", _StubPolicy)

    from testbed.config import build_policy
    build_policy({"backend": "transformers", "model_id": "m",
                  "temperature": 0.5, "top_p": 0.9, "top_k": 10})

    assert captured["top_p"] == 0.9
    assert captured["top_k"] == 10
```

This monkeypatches the class `build_policy` imports internally, so no real model is loaded — CPU-only, no network.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_and_e2e.py::test_build_policy_passes_top_p_top_k_for_transformers_backend -v`
Expected: FAIL with `KeyError: 'top_p'`

- [ ] **Step 3: Implement**

In `testbed/config.py`, change the `transformers` branch of `build_policy` (currently lines 30-35):

```python
    if backend == "transformers":
        from testbed.policy.transformers_policy import TransformersPolicy
        return TransformersPolicy(
            model_id=model_id,
            temperature=model_cfg.get("temperature", 0.7),
            top_p=model_cfg.get("top_p", 0.8),
            top_k=model_cfg.get("top_k", 20),
            steering=steering)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_and_e2e.py::test_build_policy_passes_top_p_top_k_for_transformers_backend -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add testbed/config.py tests/test_config_and_e2e.py
git commit -m "feat: pass top_p/top_k from run config into TransformersPolicy"
```

---

### Task 4: Multi-stream (resid + MLP) activation capture in `collect_activations.py`

**Files:**
- Modify: `scripts/collect_activations.py` (full rewrite of hook logic + main)
- Test: `tests/test_collect_activations.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_collect_activations.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_collect_activations.py -v`
Expected: FAIL — `ImportError: cannot import name '_resolve_streams'` (doesn't exist yet), and the existing `_extract_all_tokens` signature doesn't accept a `dict` of modules.

- [ ] **Step 3: Implement**

Replace the full contents of `scripts/collect_activations.py` with:

```python
"""Collect residual-stream and MLP activations from real multi-round game traces.

Runs game episodes with a random policy to build realistic multi-round
histories, then passes each agent's prompt through the model (with and
without the ToM suffix) and captures activations for one or more streams
(resid, mlp) at a chosen layer, in a single forward pass.

Per stream, three output files:
  activations/base_{game}_l{N}_{stream}.npy   — [T, d_model]  all token
                                        positions, base prompts (SAE training)
  activations/tom_{game}_l{N}_{stream}.npy    — [P, d_model]  last token only,
                                        ToM-suffix prompts (differential)
  activations/base_last_{game}_l{N}_{stream}.npy — [P, d_model] last token of
                                        base (paired with tom for differential)

Usage
-----
python scripts/collect_activations.py \\
    --game beauty_contest \\
    --num-episodes 200 \\
    --max-rounds 4 \\
    --layer model.layers.18 \\
    --streams resid,mlp \\
    --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from testbed.registry import build_game  # noqa: E402

TOM_SUFFIX_BEAUTY_CONTEST = (
    "\n\nBefore you answer, think carefully about what the other players "
    "are likely to do. Model their reasoning and best-respond to it."
)

TOM_SUFFIX_GBS = (
    "\n\nBefore you answer, think carefully about what the other players "
    "are likely to contribute. Coordinate with them so your contributions "
    "sum to the target as quickly as possible."
)

TOM_SUFFIX_BY_GAME = {
    "beauty_contest": TOM_SUFFIX_BEAUTY_CONTEST,
    "gbs": TOM_SUFFIX_GBS,
}


# ── random policy ─────────────────────────────────────────────────────────────

class _RandomPolicy:
    """Generates random valid actions without loading any model."""

    def act(self, system: str, user: str, agent_id: str, steering) -> str:
        # Goldstone GBS: submit a random non-negative contribution
        if "CONTRIBUTION:" in user:
            return f"CONTRIBUTION: {random.randint(0, 50)}"
        # beauty_contest: parse allowed range from the rendered user prompt
        low, high = 0, 100
        for line in user.splitlines():
            if "between" in line and "and" in line:
                parts = line.replace("(inclusive)", "").split()
                try:
                    idx = parts.index("between")
                    low = int(parts[idx + 1].strip(".,"))
                    high = int(parts[idx + 3].strip(".,"))
                except (ValueError, IndexError):
                    pass
                break
        return f"CHOICE: {random.randint(low, high)}"


# ── episode runner (no model) ─────────────────────────────────────────────────

def run_episode_collect(env, renderer, parser, policy, max_rounds: int):
    """Run one episode and return list of (system, user) prompt pairs."""
    env.reset()
    prompts = []
    done = False
    round_idx = 0

    while not done and round_idx < max_rounds:
        pending = env.pending()
        actions = {}
        for agent_id, raw_obs in pending:
            system = renderer.system_prompt(agent_id)
            user = renderer.render(raw_obs, agent_id, None)
            prompts.append((system, user))
            completion = policy.act(system, user, agent_id, None)
            result = parser.parse(completion, raw_obs, agent_id, None)
            from testbed.types import ParsedAction
            actions[agent_id] = result.value if isinstance(result, ParsedAction) else 0
        result = env.submit(actions)
        done = result.done
        round_idx += 1

    return prompts


# ── activation extraction ─────────────────────────────────────────────────────

def _resolve_submodule(model, dotted_name: str):
    obj = model
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve_streams(model, layer: str, streams: list[str]) -> dict:
    """Map each requested stream name to the submodule to hook.

    resid -> the decoder block itself (post-residual-add output, i.e. the
             standard residual stream at this layer)
    mlp   -> the block's mlp submodule (its raw output, before being summed
             back into the residual stream)
    """
    modules = {}
    for stream in streams:
        if stream == "resid":
            modules[stream] = _resolve_submodule(model, layer)
        elif stream == "mlp":
            modules[stream] = _resolve_submodule(model, layer + ".mlp")
        else:
            raise ValueError(f"Unknown stream: {stream!r} (expected 'resid' or 'mlp')")
    return modules


def _extract_all_tokens(model, tokenizer, system: str, user: str,
                        layer_modules: dict, device: str) -> dict:
    """Return all token hidden states for each stream in one forward pass.

    layer_modules: {stream_name: submodule_to_hook}
    Returns: {stream_name: Tensor[seq_len, d_model]}
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    captured = {}

    def _make_hook(stream_name):
        def hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[stream_name] = h[0].detach().float()  # [seq_len, d_model]
        return hook

    handles = [module.register_forward_hook(_make_hook(name))
               for name, module in layer_modules.items()]
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return captured


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="beauty_contest",
                    choices=["beauty_contest", "gbs"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--layer", default="model.layers.18")
    ap.add_argument("--streams", default="resid,mlp",
                    help="Comma-separated stream names to capture: resid, mlp")
    ap.add_argument("--num-episodes", type=int, default=500)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--output-dir", default="/scratch/inf0/user/nzukowsk/activations")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    streams = args.streams.split(",")

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    layer_tag = args.layer.replace(".", "_")
    out_paths = {
        stream: {
            "combined": os.path.join(
                args.output_dir, f"base_{args.game}_{layer_tag}_{stream}.npy"),
            "base_last": os.path.join(
                args.output_dir, f"base_last_{args.game}_{layer_tag}_{stream}.npy"),
            "tom_last": os.path.join(
                args.output_dir, f"tom_{args.game}_{layer_tag}_{stream}.npy"),
        }
        for stream in streams
    }

    # ── 1. generate game traces (no model needed) ────────────────────────────
    print(f"Generating {args.num_episodes} episodes "
          f"(game={args.game}, players={args.num_players}, "
          f"max_rounds={args.max_rounds}) ...")
    policy = _RandomPolicy()
    all_prompts: list[tuple[str, str]] = []

    for ep in range(args.num_episodes):
        env, renderer, parser = build_game(
            family="symbolic", game_id=args.game,
            num_players=args.num_players, env_kwargs={})
        prompts = run_episode_collect(
            env, renderer, parser, policy, args.max_rounds)
        all_prompts.extend(prompts)
        if (ep + 1) % 50 == 0:
            print(f"  {ep + 1}/{args.num_episodes} episodes "
                  f"({len(all_prompts)} prompts so far)")

    print(f"Total prompts: {len(all_prompts)}")

    # ── 2. load model ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Loading {args.model} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(device)
    model.eval()
    layer_modules = _resolve_streams(model, args.layer, streams)
    print(f"Hooked: {args.layer}  streams={streams}")

    # ── 3. count tokens first so memmaps are allocated at the exact right size ─
    tom_suffix = TOM_SUFFIX_BY_GAME[args.game]
    d_model    = model.config.hidden_size

    print("Counting tokens (no GPU)...", flush=True)
    total_tokens = 0
    for system, user in all_prompts:
        for text in (user, user + tom_suffix):
            msgs = [{"role": "system", "content": system},
                    {"role": "user",   "content": text}]
            formatted = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            total_tokens += tokenizer(formatted, return_tensors="pt")["input_ids"].shape[1]
    print(f"  total tokens: {total_tokens}  "
          f"(~{total_tokens * d_model * 4 * len(streams) / 1e9:.1f} GB on disk "
          f"across {len(streams)} stream(s))", flush=True)

    # ── 4. allocate exact-size memmaps — no rewrite needed ───────────────────
    combined_mms = {
        stream: np.lib.format.open_memmap(
            out_paths[stream]["combined"], mode="w+", dtype="float32",
            shape=(total_tokens, d_model))
        for stream in streams
    }

    base_last_lists = {stream: [] for stream in streams}
    tom_last_lists  = {stream: [] for stream in streams}
    row = 0

    for i, (system, user) in enumerate(all_prompts):
        h_base = _extract_all_tokens(model, tokenizer, system, user,
                                     layer_modules, device)
        h_tom  = _extract_all_tokens(model, tokenizer, system,
                                     user + tom_suffix, layer_modules, device)

        seq_base = h_base[streams[0]].shape[0]
        for stream in streams:
            t = h_base[stream]
            combined_mms[stream][row:row + seq_base] = \
                (t.cpu() if t.is_cuda else t).numpy()
        row += seq_base

        seq_tom = h_tom[streams[0]].shape[0]
        for stream in streams:
            t = h_tom[stream]
            combined_mms[stream][row:row + seq_tom] = \
                (t.cpu() if t.is_cuda else t).numpy()
        row += seq_tom

        for stream in streams:
            base_last_lists[stream].append(h_base[stream][-1].cpu())
            tom_last_lists[stream].append(h_tom[stream][-1].cpu())

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_prompts)} prompts processed  "
                  f"({row}/{total_tokens} rows)", flush=True)

    for stream in streams:
        combined_mms[stream].flush()
    del combined_mms

    # ── 5. save paired last-token arrays ────────────────────────────────────
    print(f"\nSaved:", flush=True)
    for stream in streams:
        base_last = torch.stack(base_last_lists[stream]).numpy()
        tom_last  = torch.stack(tom_last_lists[stream]).numpy()
        np.save(out_paths[stream]["base_last"], base_last)
        np.save(out_paths[stream]["tom_last"],  tom_last)

        print(f"  [{stream}] {out_paths[stream]['combined']}   "
              f"[{total_tokens}, {d_model}]  (SAE training: base + ToM tokens)")
        print(f"  [{stream}] {out_paths[stream]['base_last']}  {base_last.shape}  "
              f"(paired base last-token)")
        print(f"  [{stream}] {out_paths[stream]['tom_last']}   {tom_last.shape}  "
              f"(paired ToM last-token)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_collect_activations.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full fast test suite to check for regressions**

Run: `python -m pytest -q`
Expected: all non-GPU tests PASS (GPU tests skip by default)

- [ ] **Step 6: Commit**

```bash
git add scripts/collect_activations.py tests/test_collect_activations.py
git commit -m "feat: capture resid+MLP streams in one forward pass, default to Qwen3-4B"
```

---

### Task 5: Qwen3-4B default + `enable_thinking=False` in `extract_steering_vector.py`

**Files:**
- Modify: `scripts/extract_steering_vector.py:139-162` (`_last_token_hidden`), `:170` (`--model` default)
- Test: `tests/test_extract_vector_prompts.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_extract_vector_prompts.py`:

```python
import torch


class _FakeBatch(dict):
    def to(self, device):
        return self


class _RecordingTokenizer:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        return "templated"

    def __call__(self, text, return_tensors=None):
        return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3]])})


def test_last_token_hidden_disables_thinking_mode():
    import torch.nn as nn
    from scripts.extract_steering_vector import _last_token_hidden

    class _TinyLayer(nn.Module):
        def forward(self, x):
            return x

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _TinyLayer()

        def forward(self, input_ids):
            x = input_ids.float().unsqueeze(-1).expand(-1, -1, 4)
            return self.layer(x)

    model = _TinyModel()
    tok = _RecordingTokenizer()
    _last_token_hidden(model, tok, "sys", "user", model.layer, "cpu")
    assert tok.template_calls[-1]["enable_thinking"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract_vector_prompts.py::test_last_token_hidden_disables_thinking_mode -v`
Expected: FAIL with `KeyError: 'enable_thinking'`

- [ ] **Step 3: Implement**

In `scripts/extract_steering_vector.py`, change `_last_token_hidden` (currently lines 139-145):

```python
def _last_token_hidden(model, tokenizer, system: str, user: str,
                       layer_module, device: str) -> torch.Tensor:
    """Return the last-token residual-stream vector at layer_module."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)
```

And change the `--model` default (currently line 170):

```python
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract_vector_prompts.py::test_last_token_hidden_disables_thinking_mode -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_steering_vector.py tests/test_extract_vector_prompts.py
git commit -m "feat: default extract_steering_vector.py to Qwen3-4B, non-thinking mode"
```

---

### Task 6: Update the GPU-gated policy test's model id

**Files:**
- Modify: `tests/policy/test_transformers_policy.py:32-37`

- [ ] **Step 1: Update the test**

Change (currently lines 32-37):

```python
@pytest.mark.gpu
def test_transformers_policy_generates_with_qwen():
    from testbed.policy.transformers_policy import TransformersPolicy
    p = TransformersPolicy(model_id="Qwen/Qwen3-4B")
    out = p.act("You are a helpful assistant.", "Say the word 'ok'.", "player_0", None)
    assert isinstance(out, str) and len(out) > 0
```

- [ ] **Step 2: Run the fast suite to confirm nothing else broke (this test stays skipped)**

Run: `python -m pytest -q`
Expected: all non-GPU tests PASS; this test shows as `skipped`

- [ ] **Step 3: Commit**

```bash
git add tests/policy/test_transformers_policy.py
git commit -m "test: point GPU-gated policy test at Qwen3-4B"
```

---

### Task 7: Update all experiment configs to Qwen3-4B + top_p/top_k

**Files:**
- Modify: `config/run_config.yaml`, `config/exp_noop.yaml`, `config/exp_prompt_one.yaml`, `config/exp_prompt_all.yaml`, `config/exp_activation_one.yaml`, `config/exp_activation_all.yaml`, `config/gbs_exp_noop.yaml`, `config/gbs_exp_noop_directional.yaml`, `config/gbs_exp_prompt_one.yaml`, `config/gbs_exp_prompt_all.yaml`, `config/gbs_exp_prompt_all_directional.yaml`, `config/gbs_exp_activation_one.yaml`, `config/gbs_exp_activation_all.yaml`

All 13 files currently contain this identical block:

```yaml
  model_id: Qwen/Qwen2.5-3B-Instruct
  temperature: 0.7
```

- [ ] **Step 1: Edit each file**

In every one of the 13 files listed above, replace:

```yaml
  model_id: Qwen/Qwen2.5-3B-Instruct
  temperature: 0.7
```

with:

```yaml
  model_id: Qwen/Qwen3-4B
  temperature: 0.7
  top_p: 0.8
  top_k: 20
```

(`config/run_config.yaml` has an extra `endpoint:` line between `model_id` and `temperature` — match on the two lines shown, which are still contiguous and identical there.)

- [ ] **Step 2: Verify no old model id remains**

Run: `grep -rl "Qwen2.5-3B-Instruct" config/`
Expected: no output (no matches)

- [ ] **Step 3: Run the fast suite to confirm config parsing still works**

Run: `python -m pytest tests/test_config_and_e2e.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add config/
git commit -m "config: switch all experiment configs to Qwen3-4B with matched sampling params"
```

---

### Task 8: `scripts/run_layer_sweep.py` orchestrator

**Files:**
- Create: `scripts/run_layer_sweep.py`
- Test: `tests/test_run_layer_sweep.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_layer_sweep.py`:

```python
"""Unit tests for the disk-bounded layer-sweep orchestrator. All subprocess
calls are replaced with a recording fake — no real model, GPU, or training
runs in this test."""
import argparse
import os

from scripts.run_layer_sweep import run_sweep, _combined_path


def _make_args(tmp_path, **overrides):
    defaults = dict(
        game="beauty_contest", model="fake-model", streams=["resid", "mlp"],
        start_layer=10, end_layer=11, num_episodes=5, max_rounds=4, num_players=4,
        seed=42, activations_dir=str(tmp_path / "activations"),
        sae_dir=str(tmp_path / "sae"), vectors_dir=str(tmp_path / "vectors"),
        d_sae=16384, k=32, epochs=20, top_n=16, wandb=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_sweep_deletes_combined_activations_after_each_layer(tmp_path):
    args = _make_args(tmp_path)
    os.makedirs(args.activations_dir, exist_ok=True)
    calls = []

    def fake_run(cmd, check=True):
        calls.append(cmd)
        if cmd[1].endswith("collect_activations.py"):
            layer = cmd[cmd.index("--layer") + 1]
            for stream in args.streams:
                path = _combined_path(args.activations_dir, args.game, layer, stream)
                open(path, "w").close()
        return None

    run_sweep(args, run=fake_run)

    # 2 layers x (1 collect + 2 streams x 2 (train_sae + find_tom_features)) = 10
    assert len(calls) == 10
    for layer_idx in (10, 11):
        for stream in args.streams:
            path = _combined_path(args.activations_dir, args.game,
                                  f"model.layers.{layer_idx}", stream)
            assert not os.path.exists(path)


def test_run_sweep_auto_detects_end_layer(monkeypatch, tmp_path):
    args = _make_args(tmp_path, end_layer=None)
    monkeypatch.setattr("scripts.run_layer_sweep._detect_end_layer", lambda model: 10)
    calls = []
    run_sweep(args, run=lambda cmd, check=True: calls.append(cmd))
    # start=10, detected end=10 -> exactly one layer's worth: 1 + 2*2 = 5
    assert len(calls) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_layer_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_layer_sweep'`

- [ ] **Step 3: Implement**

Create `scripts/run_layer_sweep.py`:

```python
"""Sweep every layer's residual stream and MLP output through the full
extract -> train SAE -> find ToM vector pipeline, one (layer, stream) combo
at a time, deleting each combo's large raw-activation file as soon as its
SAE and steering vector are produced.

This keeps peak scratch usage to roughly one layer's worth of raw
activations (~90GB at Qwen3-4B's size for both streams) instead of
accumulating every layer simultaneously (multi-TB). The trade-off: the
model's forward pass is re-run once per swept layer instead of once total,
since a forward pass always computes every layer regardless of which one is
hooked. See docs/superpowers/specs/2026-06-16-qwen3-4b-layer-sweep-design.md.

Usage
-----
python scripts/run_layer_sweep.py \\
    --game beauty_contest --model Qwen/Qwen3-4B \\
    --start-layer 10 --num-episodes 200 --max-rounds 4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _detect_end_layer(model_id: str) -> int:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    return cfg.num_hidden_layers - 1


def _layer_tag(layer: str) -> str:
    return layer.replace(".", "_")


def _combined_path(activations_dir: str, game: str, layer: str, stream: str) -> str:
    return os.path.join(activations_dir, f"base_{game}_{_layer_tag(layer)}_{stream}.npy")


def _base_last_path(activations_dir: str, game: str, layer: str, stream: str) -> str:
    return os.path.join(activations_dir, f"base_last_{game}_{_layer_tag(layer)}_{stream}.npy")


def _tom_last_path(activations_dir: str, game: str, layer: str, stream: str) -> str:
    return os.path.join(activations_dir, f"tom_{game}_{_layer_tag(layer)}_{stream}.npy")


def _sae_path(sae_dir: str, game: str, layer: str, stream: str, d_sae: int, k: int) -> str:
    return os.path.join(sae_dir, f"{game}_{_layer_tag(layer)}_{stream}_d{d_sae}_k{k}.pt")


def _collect_command(args, layer: str) -> list[str]:
    return [sys.executable, "scripts/collect_activations.py",
            "--game", args.game, "--model", args.model, "--layer", layer,
            "--streams", ",".join(args.streams),
            "--num-episodes", str(args.num_episodes),
            "--max-rounds", str(args.max_rounds),
            "--num-players", str(args.num_players),
            "--output-dir", args.activations_dir,
            "--seed", str(args.seed)]


def _train_sae_command(args, layer: str, stream: str) -> list[str]:
    cmd = [sys.executable, "scripts/train_sae.py",
           "--activations", _combined_path(args.activations_dir, args.game, layer, stream),
           "--d-sae", str(args.d_sae), "--k", str(args.k), "--epochs", str(args.epochs),
           "--output", _sae_path(args.sae_dir, args.game, layer, stream, args.d_sae, args.k)]
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def _find_tom_features_command(args, layer: str, stream: str) -> list[str]:
    cmd = [sys.executable, "scripts/find_tom_features.py",
           "--sae", _sae_path(args.sae_dir, args.game, layer, stream, args.d_sae, args.k),
           "--base", _base_last_path(args.activations_dir, args.game, layer, stream),
           "--tom", _tom_last_path(args.activations_dir, args.game, layer, stream),
           "--top-n", str(args.top_n), "--output-dir", args.vectors_dir]
    if args.wandb:
        cmd.append("--wandb")
    return cmd


def run_sweep(args, run=subprocess.run) -> None:
    os.makedirs(args.activations_dir, exist_ok=True)
    os.makedirs(args.sae_dir, exist_ok=True)
    os.makedirs(args.vectors_dir, exist_ok=True)

    end_layer = args.end_layer
    if end_layer is None:
        end_layer = _detect_end_layer(args.model)
        print(f"Auto-detected end layer: {end_layer}")

    for layer_idx in range(args.start_layer, end_layer + 1):
        layer = f"model.layers.{layer_idx}"
        print(f"\n=== layer {layer_idx} ({layer}) ===")

        run(_collect_command(args, layer), check=True)

        for stream in args.streams:
            run(_train_sae_command(args, layer, stream), check=True)
            run(_find_tom_features_command(args, layer, stream), check=True)

            combined = _combined_path(args.activations_dir, args.game, layer, stream)
            if os.path.exists(combined):
                os.remove(combined)
                print(f"  deleted {combined}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="beauty_contest", choices=["beauty_contest", "gbs"])
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--streams", default="resid,mlp")
    ap.add_argument("--start-layer", type=int, default=10)
    ap.add_argument("--end-layer", type=int, default=None,
                    help="Default: auto-detect (model's num_hidden_layers - 1)")
    ap.add_argument("--num-episodes", type=int, default=500)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--num-players", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--activations-dir", default="/scratch/inf0/user/nzukowsk/activations")
    ap.add_argument("--sae-dir", default="/scratch/inf0/user/nzukowsk/sae")
    ap.add_argument("--vectors-dir", default="vectors")
    ap.add_argument("--d-sae", type=int, default=16384)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=16)
    ap.add_argument("--wandb", action="store_true",
                    help="Forward --wandb to train_sae.py / find_tom_features.py")
    args = ap.parse_args()
    args.streams = args.streams.split(",")
    run_sweep(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_layer_sweep.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full fast test suite to check for regressions**

Run: `python -m pytest -q`
Expected: all non-GPU tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/run_layer_sweep.py tests/test_run_layer_sweep.py
git commit -m "feat: add disk-bounded full-layer SAE sweep orchestrator"
```

---

### Task 9: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the Overview model reference**

Change (currently line 7):

```markdown
Agents powered by small local LLMs (default: `Qwen/Qwen3-4B`, non-thinking mode) play coordination and negotiation games. Steering methods are applied per-agent, and every turn is logged for behavioral analysis.
```

- [ ] **Step 2: Update the quickstart config example's model block**

Change (currently lines 54-58):

```markdown
model:
  backend: transformers     # transformers (steering-capable) | vllm (fast baseline)
  model_id: Qwen/Qwen3-4B
  endpoint: http://localhost:8000   # used only by vllm backend
  temperature: 0.7
  top_p: 0.8
  top_k: 20
```

- [ ] **Step 3: Update the Stage 1 section to document `--streams` and the new filenames**

Replace the Stage 1 section (currently lines 135-157) with:

```markdown
### Stage 1 — Collect residual-stream and MLP activations

Plays real multi-round episodes with a random policy, then runs each agent prompt through the model twice — once as-is, once with a ToM suffix appended — and captures one or more streams at the chosen layer in a single forward pass:

```bash
# Beauty contest — both streams (default)
python scripts/collect_activations.py \
  --game beauty_contest --layer model.layers.18 --streams resid,mlp \
  --num-episodes 200 --max-rounds 4 --num-players 4

# GBS
python scripts/collect_activations.py \
  --game gbs --layer model.layers.18 --streams resid,mlp \
  --num-episodes 200 --max-rounds 4 --num-players 4
```

`--streams` accepts `resid` (the decoder block's full output — the residual
stream after layer N) and/or `mlp` (the block's MLP submodule's raw output,
before it's summed into the residual stream).

Outputs saved to `--output-dir` (default `/scratch/inf0/user/nzukowsk/activations/`), one set per stream:

| File | Contents |
|------|----------|
| `base_{game}_model_layers_18_resid.npy` | All token positions, base + ToM prompts combined (SAE training set) |
| `base_last_{game}_model_layers_18_resid.npy` | Last token only, base prompts (paired with the file below) |
| `tom_{game}_model_layers_18_resid.npy` | Last token only, ToM-suffix prompts (paired) |
| *(same three, with `_mlp` instead of `_resid`)* | MLP stream |
```

- [ ] **Step 4: Add a new subsection after Stage 3 documenting the full sweep**

Insert after the existing Stage 3 section (after current line 226, before "## Experiment configs"):

```markdown
### Sweeping every layer (both streams) within a disk budget

Running Stages 1-3 by hand for every layer would require keeping every
layer's raw activations on disk simultaneously — multiple TB. Instead, run
the sweep orchestrator, which processes one (layer, stream) combo fully
before moving to the next, deleting the large raw-activation file
immediately after its SAE and steering vector are produced:

```bash
python scripts/run_layer_sweep.py \
  --game beauty_contest --model Qwen/Qwen3-4B \
  --start-layer 10 --num-episodes 200 --max-rounds 4
```

`--end-layer` defaults to the model's last layer (auto-detected). Peak
scratch usage stays around one layer's worth of raw activations for both
streams (tens of GB) regardless of how many layers are swept — the
trade-off is that the model's forward pass is re-run once per swept layer
instead of once, since a forward pass always computes every layer
regardless of which one is hooked. SAE checkpoints, steering vectors, and
the small last-token files persist for every layer; only the large
all-token-position files are deleted.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document Qwen3-4B default and the full-layer SAE sweep"
```

---

### Task 10: Final full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full fast test suite**

Run: `python -m pytest -q`
Expected: all tests pass (GPU tests show as `skipped`)

- [ ] **Step 2: Confirm no stray references to the old model id remain**

Run: `grep -rn "Qwen2.5-3B-Instruct" --include=*.py --include=*.yaml --include=*.md .`
Expected: no output

- [ ] **Step 3: Report status to the user**

Summarize: all tasks committed, full pipeline updated to Qwen3-4B with matched sampling, multi-stream resid+MLP capture implemented, layer-sweep orchestrator in place, README updated. Remind the user that running the actual sweep (and `wandb login` if they want logging) happens on their cluster, not this machine.
