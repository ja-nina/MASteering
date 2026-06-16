# Qwen3-4B Migration + Full-Layer SAE Sweep — Design

**Date:** 2026-06-16
**Status:** Approved design, ready for implementation planning

## 1. Purpose

Two related changes to the SAE/steering pipeline described in
[2026-06-14-llm-steering-multigame-testbed-design.md](2026-06-14-llm-steering-multigame-testbed-design.md):

1. Swap the base model from `Qwen/Qwen2.5-3B-Instruct` to `Qwen/Qwen3-4B`,
   run in non-thinking mode, with sampling parameters matching Qwen's
   official non-thinking recommendation.
2. Extend the SAE activation-extraction pipeline (currently: one residual-stream
   layer, picked by hand) to sweep **every layer from 10 to the last, capturing
   both the residual stream and the MLP output at each layer**, while keeping
   raw-activation disk usage bounded to roughly one layer's worth at a time
   (hard constraint: 1TB of cluster scratch space).

Code changes only — this machine (GTX 1650, 4.3GB VRAM) cannot fit Qwen3-4B.
The actual sweep runs on the user's cluster; this work produces correct,
ready-to-run code and docs.

## 2. Model + sampling changes

- Replace `Qwen/Qwen2.5-3B-Instruct` with `Qwen/Qwen3-4B` as the default
  `model_id` in: `config/run_config.yaml`, all 12 `config/*.yaml` experiment
  configs, and the `--model` default in `collect_activations.py` and
  `extract_steering_vector.py`.
- Thread `enable_thinking=False` into every `tokenizer.apply_chat_template(...)`
  call so Qwen3 never emits `<think>` blocks (which would otherwise pollute
  both generated actions and captured activations). Call sites:
  `TransformersPolicy._build_inputs`, `collect_activations.py`'s
  `_extract_all_tokens`, `extract_steering_vector.py`'s `_last_token_hidden`.
  Passing this kwarg to a template that doesn't reference it (e.g. Qwen2.5) is
  a harmless no-op.
- Add `top_p` / `top_k` support to `TransformersPolicy._generate` and
  `build_policy` (new config keys `model.top_p`, `model.top_k`), defaulting to
  `0.8` / `20` — Qwen3's documented non-thinking sampling profile. Temperature
  stays `0.7` (already correct in every config). Add both keys to all 13
  config YAML files. `VLLMPolicy` is untouched — it's not part of the
  steering/SAE pipeline and wasn't part of this request.

## 3. Multi-stream activation capture

`collect_activations.py` currently hooks one dotted submodule path
(`model.layers.N`, i.e. the residual stream after block N) and writes one set
of three files (`base_`, `base_last_`, `tom_`). Changes:

- New `--streams` argument, default `"resid,mlp"`. For a given `--layer
  model.layers.N`: `resid` hooks `model.layers.N` (current behavior — the
  decoder block's full output, i.e. post-residual-add); `mlp` hooks
  `model.layers.N.mlp` (the MLP submodule's raw output, before it's summed
  into the residual stream — the standard mechanistic-interpretability
  "mlp_out" hook point). There is currently no MLP-specific extraction
  anywhere in the codebase; this is new.
- `_extract_all_tokens` registers **all requested hooks before a single
  `model(**inputs)` call** and returns `dict[stream_name, Tensor]`, so
  resid+mlp for one layer cost exactly one forward pass, not two.
- Output filenames gain the stream name:
  `base_{game}_{layer_tag}_{stream}.npy`, `base_last_{game}_{layer_tag}_{stream}.npy`,
  `tom_{game}_{layer_tag}_{stream}.npy`. This renames today's resid-only output
  (`base_{game}_l18.npy` style) slightly; acceptable since these are
  regenerated scratch artifacts, never checked into the repo.
- `train_sae.py` and `find_tom_features.py` need no code changes — both are
  already filename-agnostic, taking explicit `--activations` / `--sae` /
  `--base` / `--tom` paths.

## 4. Layer-sweep orchestrator (disk-bounded)

New script: `scripts/run_layer_sweep.py`.

For each game (`beauty_contest`, `gbs`) and each layer `L` from `--start-layer`
(default `10`) to `--end-layer` (default: auto-detected as
`AutoConfig.from_pretrained(model).num_hidden_layers - 1`, i.e. `35` for
Qwen3-4B):

1. Subprocess: `collect_activations.py --layer model.layers.{L} --streams resid,mlp ...`
2. For each stream in `{resid, mlp}`:
   a. Subprocess: `train_sae.py --activations <combined file> --output <sae dir>/...`
   b. Subprocess: `find_tom_features.py --sae <ckpt> --base <base_last> --tom <tom_last> ...`
3. Delete the two large `base_{game}_{layer_tag}_{stream}.npy` combined files
   for this layer (the ones holding every token position). Keep
   `base_last_*` / `tom_last_*` (last-token only, small), the SAE checkpoint
   (~336MB), and the steering vector + CSV (tiny).
4. Proceed to layer `L+1`.

Each stage runs as a subprocess (not an in-process function call) so GPU
memory is fully released between stages — stage 1 needs the LM resident,
stages 2-3 don't — and so each script's existing, independently-tested CLI
contract is reused unchanged.

**Disk:** peak usage is bounded to ~2 streams' worth of raw activations at any
point (~90GB at Qwen3-4B's `d_model=2560`, extrapolated from the documented
34GB-per-stream figure at Qwen2.5-3B's `d_model=2048` and the same episode
settings), regardless of how many layers are swept. Comfortably inside the
1TB scratch budget.

**Compute trade-off (explicitly accepted):** a transformer forward pass
computes every layer's activations regardless of which one is hooked, so
re-extracting per layer means the model's forward cost is paid once per swept
layer (~26x for layers 10-35) instead of once. This is the necessary cost of
keeping disk bounded under the 1TB ceiling rather than capturing all layers in
one pass and storing them simultaneously (~4TB, over budget).

`run_layer_sweep.py` passes through episode/training hyperparameters
(`--num-episodes`, `--max-rounds`, `--num-players`, `--d-sae`, `--k`,
`--epochs`, `--top-n`) to the relevant sub-stage, plus an optional `--wandb`
flag forwarded to `train_sae.py` / `find_tom_features.py`.

## 5. wandb

No change to logging structure. Two existing projects: `ma-steering`
(episode runs, gated by `wandb.enabled: true` in the run config) and
`ma-steering-sae` (SAE training + ToM feature finding, gated by passing
`--wandb`). Both are opt-in; `run_layer_sweep.py` only logs if `--wandb` is
passed through. Whichever machine runs the sweep needs `wandb` installed and
a one-time `wandb login` (or `WANDB_API_KEY` env var) under the account that
owns those projects — out of scope for this change, the user's responsibility.

## 6. Testing

- CPU-only unit test for multi-hook capture: a tiny dummy `nn.Module` with a
  `.mlp` submodule, asserting one `model(...)` call populates both stream
  results via hooks registered together — follows the existing `TinyModel`
  pattern in `tests/policy/test_transformers_policy.py`.
- CPU-only unit test asserting `top_p` / `top_k` are passed into the
  `model.generate(...)` call args in `TransformersPolicy._generate`.
- Update the one `@pytest.mark.gpu` test's `model_id` to `Qwen/Qwen3-4B` (skips
  on this CPU/low-VRAM machine; correctness matters for whoever runs it with
  `TESTBED_FORCE_GPU_TESTS=1` on the cluster).

## 7. Docs

Update `README.md`: Quickstart model reference, Stage 1 section (new
`--streams` flag and renamed output files), and a new subsection documenting
`run_layer_sweep.py` as the recommended way to build the full layer×stream
sweep, including the disk-budget trade-off called out inline.

## 8. Out of scope

- Running the sweep itself (no cluster access from this machine).
- vLLM backend changes.
- Online/streaming SAE training (would avoid the disk vs. compute trade-off
  in §4 entirely, but is a much larger architecture change than asked for).
- wandb installation/login (cluster-side, user's one-time step).
