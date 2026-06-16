# LLM Steering Multi-Game Testbed

A research testbed for measuring how **steering vectors** and **prompt injections** change LLM agent behavior across text-based multi-agent games.

## Overview

Agents powered by small local LLMs (default: `Qwen/Qwen2.5-3B-Instruct`) play coordination and negotiation games. Steering methods are applied per-agent, and every turn is logged for behavioral analysis.

**Games supported**

| Family | ID | Description |
|--------|----|-------------|
| `symbolic` | `beauty_contest` | N players guess an integer; winner is closest to 2/3 of the group average |
| `symbolic` | `gbs` | Goldstone Group Sum — players each submit a contribution; the group wins when contributions sum to a hidden target. Feedback is `exact` (signed error, e.g. "too HIGH by 23") or `directional` ("too HIGH" only) |
| `textarena` | any TextArena env ID | Turn-based text games (Taboo, etc.) via the TextArena library |

**Steering methods**

| Method | Description |
|--------|-------------|
| `noop` | Baseline — no steering applied |
| `prompt_injection` | Per-agent system suffix / user prefix injected at inference time |
| `activation` | Adds a pre-computed vector (`.npy` / `.pt`) to a named residual-stream layer via a forward hook |

## Quickstart

```bash
# install dependencies
pip install -r requirements.txt

# run one episode with the default config (beauty_contest, 4 players, noop steering)
python scripts/run_episode.py --config config/run_config.yaml
```

Episode logs land in `logs/<run_id>/episode_N.jsonl` with a `.summary.json` sidecar.

## Configuration

Edit `config/run_config.yaml` or pass a custom YAML:

```yaml
run_id: my_run

game:
  family: symbolic          # symbolic | textarena
  id: beauty_contest        # beauty_contest | gbs | <TextArena env_id>
  env_kwargs:
    num_rounds: 5
    # gbs only:
    # feedback: exact       # exact | directional

episodes: 1

model:
  backend: transformers     # transformers (steering-capable) | vllm (fast baseline)
  model_id: Qwen/Qwen2.5-3B-Instruct
  endpoint: http://localhost:8000   # used only by vllm backend
  temperature: 0.7

agents:
  count: 4
  concurrency: sequential
  max_parse_retries: 5

steering:
  default: noop             # noop | prompt_injection | activation
  per_agent: {}

logging:
  dir: logs/
```

### Prompt injection — one agent

Inject a text prefix/suffix into one player's prompt only:

```yaml
steering:
  default: prompt_injection
  per_agent:
    player_0:
      user_prefix: >-
        Before answering, think carefully about what the other players are
        likely to do. Model their reasoning and best-respond to it.
```

### Prompt injection — all agents

Use `default_config` to apply the same injection to every agent not listed in `per_agent`:

```yaml
steering:
  default: prompt_injection
  default_config:
    user_prefix: >-
      Before answering, think carefully about what the other players are
      likely to do. Model their reasoning and best-respond to it.
  per_agent: {}
```

### Activation steering — one agent

Obtain a steering vector (e.g., via the SAE pipeline below) and save it as a `.npy` file, then:

```yaml
steering:
  default: activation
  per_agent:
    player_0:
      layer: model.layers.18     # dotted submodule path in the HF model
      vector_path: vectors/tom_sae_top16_beauty_contest_model_layers_18_d16384_k32.npy
      coefficient: 20.0
```

The vector is added to the residual stream of `player_0`'s inference only; all other players are unsteered.

### Activation steering — all agents

```yaml
steering:
  default: activation
  default_config:
    layer: model.layers.18
    vector_path: vectors/tom_sae_top16_beauty_contest_model_layers_18_d16384_k32.npy
    coefficient: 20.0
  per_agent: {}
```

## Extracting a Theory-of-Mind steering vector

The steering vectors in `config/*_exp_activation_*.yaml` are extracted via a three-stage SAE pipeline. Run the stages in order once before launching activation-steered experiments.

Large intermediate artifacts (raw activations, SAE checkpoints) default to a scratch path; steering vectors are small and stay in the local repo under `vectors/`.

### Stage 1 — Collect residual-stream activations

Plays real multi-round episodes with a random policy, then runs each agent prompt through the model twice — once as-is, once with a ToM suffix appended — and captures the residual stream at the chosen layer:

```bash
# Beauty contest
python scripts/collect_activations.py \
  --game beauty_contest --layer model.layers.18 \
  --num-episodes 200 --max-rounds 4 --num-players 4

# GBS
python scripts/collect_activations.py \
  --game gbs --layer model.layers.18 \
  --num-episodes 200 --max-rounds 4 --num-players 4
```

Outputs saved to `--output-dir` (default `/scratch/inf0/user/nzukowsk/activations/`):

| File | Contents |
|------|----------|
| `base_{game}_model_layers_18.npy` | All token positions, base + ToM prompts combined (SAE training set) |
| `base_last_{game}_model_layers_18.npy` | Last token only, base prompts (paired with the file below) |
| `tom_{game}_model_layers_18.npy` | Last token only, ToM-suffix prompts (paired) |

### Stage 2 — Train a TopK sparse autoencoder

Trains on the combined base+ToM activations file from Stage 1, so the dictionary captures both distributions:

```bash
# Beauty contest
python scripts/train_sae.py \
  --activations /scratch/inf0/user/nzukowsk/activations/base_beauty_contest_model_layers_18.npy \
  --d-sae 16384 --k 32 --epochs 20

# GBS
python scripts/train_sae.py \
  --activations /scratch/inf0/user/nzukowsk/activations/base_gbs_model_layers_18.npy \
  --d-sae 16384 --k 32 --epochs 20
```

If the activations file is too large to fit in memory, cap how many rows are read (reads sequentially from disk, never memory-maps the full file):

```bash
python scripts/train_sae.py \
  --activations /scratch/inf0/user/nzukowsk/activations/base_gbs_model_layers_18.npy \
  --max-samples 500000
```

Trained SAE checkpoints are saved to `--output` (default `/scratch/inf0/user/nzukowsk/sae/{stem}_d{d_sae}_k{k}.pt`):

```
/scratch/inf0/user/nzukowsk/sae/beauty_contest_model_layers_18_d16384_k32.pt
/scratch/inf0/user/nzukowsk/sae/gbs_model_layers_18_d16384_k32.pt
```

Training reports both train and held-out validation loss (10% split) each epoch, plus the percentage of dead features. Dead-feature resampling fires every `--resample-interval` epochs (default 5) while dead features exceed 1% of the dictionary, and stops after `--resample-until` epochs (default 10, i.e. the first half of the default 20-epoch run) so the second half converges cleanly.

### Stage 3 — Find ToM features and build the steering vector

```bash
# Beauty contest
python scripts/find_tom_features.py \
  --sae /scratch/inf0/user/nzukowsk/sae/beauty_contest_model_layers_18_d16384_k32.pt \
  --base /scratch/inf0/user/nzukowsk/activations/base_last_beauty_contest_model_layers_18.npy \
  --tom  /scratch/inf0/user/nzukowsk/activations/tom_beauty_contest_model_layers_18.npy \
  --top-n 16

# GBS
python scripts/find_tom_features.py \
  --sae /scratch/inf0/user/nzukowsk/sae/gbs_model_layers_18_d16384_k32.pt \
  --base /scratch/inf0/user/nzukowsk/activations/base_last_gbs_model_layers_18.npy \
  --tom  /scratch/inf0/user/nzukowsk/activations/tom_gbs_model_layers_18.npy \
  --top-n 16
```

Outputs saved to `vectors/` (local repo):

```
vectors/tom_sae_top16_beauty_contest_model_layers_18_d16384_k32.npy   ← steering vector
vectors/tom_features_beauty_contest_model_layers_18_d16384_k32.csv    ← feature scores
vectors/tom_sae_top16_gbs_model_layers_18_d16384_k32.npy
vectors/tom_features_gbs_model_layers_18_d16384_k32.csv
```

The steering vector is a unit-normalised weighted sum of the top-N SAE decoder columns, ranked by `mean(tom_acts) - mean(base_acts)` per feature.

> **No SAE?** You can also extract a steering vector directly via Contrastive Activation Addition (CAA), without training an SAE:
> ```bash
> python scripts/extract_steering_vector.py --game beauty_contest --layer model.layers.18 --num-samples 64
> python scripts/extract_steering_vector.py --game gbs --layer model.layers.18 --num-samples 64
> ```
> This computes `mean(h_ToM) - mean(h_base)` at the chosen layer and saves it to `vectors/`. The SAE route is preferred because it isolates ToM-specific directions in sparse feature space rather than mixing in unrelated variance.

## Experiment configs

All ready-to-run configs live in `config/`. Naming convention: `[game_]exp_[method]_[scope][_directional].yaml`.

| Config file | Game | Steering | Scope | Feedback |
|-------------|------|----------|-------|----------|
| `run_config.yaml` | beauty contest | noop | — | — |
| `exp_noop.yaml` | beauty contest | noop | control | — |
| `exp_prompt_one.yaml` | beauty contest | prompt injection | player_0 only | — |
| `exp_prompt_all.yaml` | beauty contest | prompt injection | all agents | — |
| `exp_activation_one.yaml` | beauty contest | activation (SAE vector) | player_0 only | — |
| `exp_activation_all.yaml` | beauty contest | activation (SAE vector) | all agents | — |
| `gbs_exp_noop.yaml` | GBS | noop | control | exact |
| `gbs_exp_noop_directional.yaml` | GBS | noop | control | directional |
| `gbs_exp_prompt_one.yaml` | GBS | prompt injection | player_0 only | exact |
| `gbs_exp_prompt_all.yaml` | GBS | prompt injection | all agents | exact |
| `gbs_exp_prompt_all_directional.yaml` | GBS | prompt injection | all agents | directional |
| `gbs_exp_activation_one.yaml` | GBS | activation (SAE vector) | player_0 only | exact |
| `gbs_exp_activation_all.yaml` | GBS | activation (SAE vector) | all agents | exact |

### Full experiment sequence

After the vectors are in place, run the experimental conditions:

```bash
# --- Beauty contest ---
python scripts/run_episode.py --config config/exp_noop.yaml
python scripts/run_episode.py --config config/exp_prompt_one.yaml
python scripts/run_episode.py --config config/exp_prompt_all.yaml
python scripts/run_episode.py --config config/exp_activation_one.yaml
python scripts/run_episode.py --config config/exp_activation_all.yaml

# --- GBS (exact feedback) ---
python scripts/run_episode.py --config config/gbs_exp_noop.yaml
python scripts/run_episode.py --config config/gbs_exp_prompt_one.yaml
python scripts/run_episode.py --config config/gbs_exp_prompt_all.yaml
python scripts/run_episode.py --config config/gbs_exp_activation_one.yaml
python scripts/run_episode.py --config config/gbs_exp_activation_all.yaml

# --- GBS (directional feedback — ToM matters most here) ---
python scripts/run_episode.py --config config/gbs_exp_noop_directional.yaml
python scripts/run_episode.py --config config/gbs_exp_prompt_all_directional.yaml
```

Each run writes `logs/<run_id>/episode_N.jsonl`, `episode_N.summary.json`, and a human-readable `episode_N.trace.txt` showing the full prompt/completion/action for every player every turn.

## Project structure

```
testbed/
  types.py                  # shared dataclasses (StepResult, SteeringSpec, …)
  orchestrator.py           # game-agnostic episode loop
  registry.py               # game_id → (adapter, renderer, parser)
  config.py                 # YAML parsing + builder functions
  envs/
    adapter.py              # EnvAdapter Protocol
    symbolic/               # BeautyContestAdapter, GBSAdapter
    textarena/               # TextArenaAdapter
  renderers/                # state → system/user prompt text
  parsers/                  # text → action, with error feedback
  steering/                 # NoOpSteering, PromptInjectionSteering, ActivationSteering
  policy/
    base.py                 # Policy Protocol + StubPolicy
    transformers_policy.py  # in-process HF inference (activation steering supported)
    vllm_policy.py          # OpenAI-compatible vLLM client (fast baselines)
  logging_/
    episode_logger.py       # JSONL + .trace.txt human-readable logs
config/                     # ready-to-run experiment configs
scripts/
  run_episode.py            # episode runner CLI
  collect_activations.py    # Stage 1: residual-stream activation collection
  train_sae.py              # Stage 2: TopK sparse autoencoder training
  find_tom_features.py      # Stage 3: ToM feature extraction + steering vector
  extract_steering_vector.py  # CAA shortcut (no SAE)
vectors/                    # .npy steering vectors produced by find_tom_features.py / extract_steering_vector.py
tests/                      # pytest suite (GPU-gated tests skip on CPU-only envs)
```

`activations/` (Stage 1 output) and `sae/` (Stage 2 output) are not part of the repo — they default to `/scratch/inf0/user/nzukowsk/` since raw activation matrices and SAE checkpoints can run into tens of GB. Override with `--output-dir` / `--output` to use a different location.

## Tests

```bash
python -m pytest -q
```

Tests that require a GPU and a modern `transformers` install are marked `@pytest.mark.gpu` and skip automatically on CPU-only environments. To force-run them:

```bash
TESTBED_FORCE_GPU_TESTS=1 python -m pytest -q
```

## Adding a new game

1. Implement `EnvAdapter` in `testbed/envs/` (subclass `SymbolicAdapter` for simultaneous games or write a wrapper for turn-based libraries).
2. Implement `TextRenderer` in `testbed/renderers/`.
3. Implement `ActionParser` in `testbed/parsers/`.
4. Register the triple in `testbed/registry.py`.

No other files need to change.
