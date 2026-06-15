# LLM Steering Multi-Game Testbed

A research testbed for measuring how **steering vectors** and **prompt injections** change LLM agent behavior across text-based multi-agent games.

## Overview

Agents powered by small local LLMs (default: `Qwen/Qwen2.5-3B-Instruct`) play coordination and negotiation games. Steering methods are applied per-agent, and every turn is logged for behavioral analysis.

**Games supported**

| Family | ID | Description |
|--------|----|-------------|
| `symbolic` | `beauty_contest` | N players guess an integer; winner is closest to 2/3 of the mean |
| `symbolic` | `gbs` | Group Binary Search — players bracket a hidden target via median feedback |
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

The steering vectors in `config/exp_activation_*.yaml` are extracted via a three-stage SAE pipeline. Run the stages in order once before launching activation-steered experiments.

### Stage 1 — Collect residual-stream activations

Run both games with a random policy and collect activations at layer 18, both with and without the ToM suffix:

```bash
# Beauty contest
python scripts/collect_activations.py --game beauty_contest --layer 18 --episodes 200

# GBS
python scripts/collect_activations.py --game gbs --layer 18 --episodes 200
```

Outputs saved to `activations/`:

| File | Contents |
|------|----------|
| `base_beauty_contest_l18.npy` | All-token activations for SAE training |
| `base_last_beauty_contest_l18.npy` | Last-token, no ToM suffix (paired) |
| `tom_last_beauty_contest_l18.npy` | Last-token, with ToM suffix (paired) |
| *(same three files for `gbs`)* | |

### Stage 2 — Train a TopK sparse autoencoder

```bash
# Beauty contest
python scripts/train_sae.py \
  --data activations/base_beauty_contest_l18.npy \
  --d_sae 4096 --k 32 --epochs 20

# GBS
python scripts/train_sae.py \
  --data activations/base_gbs_l18.npy \
  --d_sae 4096 --k 32 --epochs 20
```

Trained SAE checkpoints are saved to `sae/`:

```
sae/base_beauty_contest_l18_d16384_k32.pt
sae/base_gbs_l18_d16384_k32.pt
```

### Stage 3 — Find ToM features and build the steering vector

```bash
# Beauty contest
python scripts/find_tom_features.py \
  --sae sae/base_beauty_contest_l18_d16384_k32.pt \
  --base activations/base_last_beauty_contest_l18.npy \
  --tom  activations/tom_last_beauty_contest_l18.npy \
  --top_n 16

# GBS
python scripts/find_tom_features.py \
  --sae sae/base_gbs_l18_d16384_k32.pt \
  --base activations/base_last_gbs_l18.npy \
  --tom  activations/tom_last_gbs_l18.npy \
  --top_n 16
```

Outputs saved to `vectors/`:

```
vectors/tom_sae_top16_beauty_contest_model_layers_18_d16384_k32.npy   ← steering vector
vectors/tom_features_beauty_contest_model_layers_18_d16384_k32.csv    ← feature scores
vectors/tom_sae_top16_gbs_model_layers_18_d16384_k32.npy
vectors/tom_features_gbs_model_layers_18_d16384_k32.csv
```

The steering vector is a unit-normalised weighted sum of the top-N SAE decoder columns ranked by `mean(tom_acts) - mean(base_acts)`.

> **No SAE?** You can also extract a steering vector directly via Contrastive Activation Addition (CAA) without training an SAE:
> ```bash
> python scripts/extract_steering_vector.py --game beauty_contest --layer 18
> python scripts/extract_steering_vector.py --game gbs --layer 18
> ```
> This computes `mean(h_ToM) - mean(h_base)` at layer 18 and saves it to `vectors/`. The SAE route is preferred because it isolates ToM-specific directions in sparse feature space.

## Experiment configs

All 11 ready-to-run configs live in `config/`. Naming convention: `[game_]exp_[method]_[scope].yaml`.

| Config file | Game | Steering | Scope |
|-------------|------|----------|-------|
| `run_config.yaml` | beauty contest | noop | — |
| `exp_noop.yaml` | beauty contest | noop | control |
| `exp_prompt_one.yaml` | beauty contest | prompt injection | player_0 only |
| `exp_prompt_all.yaml` | beauty contest | prompt injection | all agents |
| `exp_activation_one.yaml` | beauty contest | activation (SAE vector) | player_0 only |
| `exp_activation_all.yaml` | beauty contest | activation (SAE vector) | all agents |
| `gbs_exp_noop.yaml` | GBS | noop | control |
| `gbs_exp_prompt_one.yaml` | GBS | prompt injection | player_0 only |
| `gbs_exp_prompt_all.yaml` | GBS | prompt injection | all agents |
| `gbs_exp_activation_one.yaml` | GBS | activation (SAE vector) | player_0 only |
| `gbs_exp_activation_all.yaml` | GBS | activation (SAE vector) | all agents |

### Full experiment sequence

After the vectors are in place, run all 10 experimental conditions:

```bash
# --- Beauty contest ---
python scripts/run_episode.py --config config/exp_noop.yaml
python scripts/run_episode.py --config config/exp_prompt_one.yaml
python scripts/run_episode.py --config config/exp_prompt_all.yaml
python scripts/run_episode.py --config config/exp_activation_one.yaml
python scripts/run_episode.py --config config/exp_activation_all.yaml

# --- GBS ---
python scripts/run_episode.py --config config/gbs_exp_noop.yaml
python scripts/run_episode.py --config config/gbs_exp_prompt_one.yaml
python scripts/run_episode.py --config config/gbs_exp_prompt_all.yaml
python scripts/run_episode.py --config config/gbs_exp_activation_one.yaml
python scripts/run_episode.py --config config/gbs_exp_activation_all.yaml
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
    textarena/              # TextArenaAdapter
  renderers/                # state → system/user prompt text
  parsers/                  # text → action, with error feedback
  steering/                 # NoOpSteering, PromptInjectionSteering, ActivationSteering
  policy/
    base.py                 # Policy Protocol + StubPolicy
    transformers_policy.py  # in-process HF inference (activation steering supported)
    vllm_policy.py          # OpenAI-compatible vLLM client (fast baselines)
  logging_/
    episode_logger.py       # JSONL + .trace.txt human-readable logs
config/                     # 11 ready-to-run experiment configs
scripts/
  run_episode.py            # episode runner CLI
  collect_activations.py    # Stage 1: residual-stream activation collection
  train_sae.py              # Stage 2: TopK sparse autoencoder training
  find_tom_features.py      # Stage 3: ToM feature extraction + steering vector
  extract_steering_vector.py  # CAA shortcut (no SAE)
activations/                # .npy files produced by collect_activations.py
sae/                        # .pt checkpoints produced by train_sae.py
vectors/                    # .npy steering vectors produced by find_tom_features.py
tests/                      # pytest suite (GPU-gated tests skip on CPU-only envs)
```

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
