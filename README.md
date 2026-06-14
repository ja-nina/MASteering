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

### Prompt injection example

```yaml
steering:
  default: prompt_injection
  per_agent:
    player_0:
      system_suffix: " Always pick the lowest number."
    player_1:
      user_prefix: "Hint: think about equilibrium. "
```

### Activation steering example

Obtain a steering vector (e.g., via contrastive activation addition) and save it as a `.npy` or `.pt` file, then:

```yaml
steering:
  default: activation
  per_agent:
    player_0:
      layer: model.layers.14     # dotted submodule path in the HF model
      vector_path: vectors/cooperative.npy
      coefficient: 20.0
```

The vector is added to the residual stream of `player_0`'s inference only; all other players are unsteered.

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
    episode_logger.py       # JSONL turn logs + summary sidecar
config/
  run_config.yaml           # default run configuration
scripts/
  run_episode.py            # CLI entry point
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
