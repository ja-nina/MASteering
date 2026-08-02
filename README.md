# LLM Steering Multi-Game Testbed

A research testbed for measuring how **activation steering** changes LLM agent behaviour across text-based multi-agent games.

## Overview

Agents powered by local LLMs play coordination and social-deduction games. Steering vectors derived from contrastive activation differences are injected per-agent at a chosen residual-stream layer. Every turn is logged for behavioural analysis.

**Games supported**

| Family | ID | Players | Status | Doc |
|--------|----|---------|--------|-----|
| `symbolic` | `avalon` | 5–10 | Skeleton — parser+renderer done, submit() pending | [avalon.md](docs/games/avalon.md) |
| `symbolic` | `hanabi` | 2–5 | Skeleton — parser+renderer done, submit() pending | [hanabi.md](docs/games/hanabi.md) |
| `symbolic` | `werewolf` | 4–10 | Skeleton — parser+renderer done, submit() pending | [werewolf.md](docs/games/werewolf.md) |
| `symbolic` | `overcooked` | 2 | Skeleton — parser+renderer done, grid sim pending | [overcooked.md](docs/games/overcooked.md) |
| `textarena` | any TextArena `env_id` | varies | Fully playable — no game code needed | [textarena.md](docs/games/textarena.md) |

**Steering methods**

| Method | Description |
|--------|-------------|
| `noop` | Baseline — no steering applied |
| `activation` | Adaptive residual-stream vector injection at a chosen decoder layer |
| `role_aware_activation` | Activation steering that re-targets agents by in-game role each episode |

## Quickstart

```bash
pip install -r requirements.txt

# Run one episode (Debate baseline)
python scripts/run_episode.py \
  --config config/games/debate/debate_noop_2p.yaml \
  --episodes 1

# Run one episode with persona vector steering on the Mafia player
export PERSONA_VECTORS_ROOT=/path/to/PersVecGen/data/vector_extraction/persona/qwen3-14b
python scripts/run_episode.py \
  --config config/games/mafia/mafia_activation_wolf_opportunistic_8p.yaml \
  --episodes 1
```

Episode logs land in `logs/<game>/<run_id>/episode_N.jsonl` with a `.summary.json` sidecar.

## Repository layout

```
testbed/                        # Core library — game-agnostic
  orchestrator.py               # Episode loop (parallel agent calls via ThreadPoolExecutor)
  registry.py                   # Maps game_id → (Adapter, Renderer, Parser)
  envs/
    symbolic/                   # Turn-based symbolic games (avalon, hanabi, werewolf, overcooked)
    textarena/
      ta_adapter.py             # Wraps the TextArena library
  renderers/symbolic/           # State → prompt text, one file per game
  parsers/symbolic/             # Text → action, one file per game
  steering/
    noop.py                     # Baseline (no steering)
    activation.py               # Adaptive / additive residual-stream vector injection
    role_aware.py               # Per-episode role-targeted activation steering
  policy/
    transformers_policy.py      # In-process HuggingFace inference
    vllm_policy.py              # OpenAI-compatible vLLM client
  logging_/episode_logger.py

config/
  shared/                       # Persona lists shared across games
    personas.yaml               # 20 character personas (Riedl 2025)
    behavioral_personas.yaml    # 34 behavioral archetypes
    run_config.yaml             # Annotated config template
  games/                        # One directory per game
    debate/
    mafia/
    negotiation/
    ...

scripts/
  run_episode.py                # Game-agnostic episode runner (primary CLI)
  textarena/                    # Per-game plot scripts
    plot_debate_results.py
    plot_mafia_results.py
    ...
  slurm/                        # SLURM array job scripts

docs/games/                     # Per-game documentation
  textarena.md
  adding_a_new_game.md          # Step-by-step guide

logs/                           # Runtime outputs (gitignored)
plots/                          # Analysis figures
```

## Persona vector steering

Precomputed persona steering vectors from [PersVecGen](../PersVecGen/) can be
injected into any `TransformersPolicy` run. Vectors are derived from
contrastive activation differences (mean positive − mean negative hidden
states) and stored as PyTorch `.pt` files keyed by layer index.

### Vector location

```bash
export PERSONA_VECTORS_ROOT=/path/to/PersVecGen/data/vector_extraction/persona/qwen3-14b
```

| Environment | Path |
|-------------|------|
| Local (Windows) | `C:\Users\ismyn\UNI\MPI\Thesis\PersVecGen\data\vector_extraction\persona\qwen3-14b` |
| Cluster | `/scratch/inf0/user/nzukowsk/PEVectors/PEVectors/data/vector_extraction/persona/qwen3-14b` |

Each trait has two precision directories (`bf16/` and `4bit/`), each containing
`<trait>.pt` (vectors for layers 10 / 20 / 29) and `<trait>.json` (extraction
metadata with per-layer sweep scores). Use `bf16/` for standard inference.

### Steering mode

The default and only mode used in experiments is **adaptive**: per-token soft
top-up toward `coefficient × ‖v‖`, never overshooting. This is the same formula
as PersVecGen's `steering.py`. Additive mode (`mode: additive` in YAML) is
available for ablations only.

### Layer auto-selection

When a config omits the `layer` key, the companion `.json` is read and the
layer with the highest `max_score − baseline_score` is selected automatically.
The layer index is mapped to a submodule path via `layer_path_template`
(default `"model.layers.{}"`, correct for Qwen3-14B and most HF decoders).

### Static steering (`activation`)

Steer a fixed set of agents every episode. Example — steer the AGAINST side
in Debate with the `assertive` vector:

```yaml
steering:
  default: activation
  per_agent:
    player_1:
      vector_path: ${PERSONA_VECTORS_ROOT}/bf16/assertive.pt
      coefficient: 1.25
      layer_path_template: "model.layers.{}"
```

Config: `config/games/debate/debate_activation_p1_assertive_2p.yaml`

### Role-aware steering (`role_aware_activation`)

Steer agents by in-game role, re-discovered after each `env.reset()`. The
steered seat changes episode-to-episode as the game assigns roles randomly.

Set `target_roles` to match the role strings that specific game engine reports:

| Game | Role string(s) |
|------|---------------|
| `SecretMafia-v0` | `mafia` |
| `Werewolf-v0` | `werewolf` |
| `Avalon-v0` | `evil`, `morgana`, `mordred` |

```yaml
steering:
  default: role_aware_activation
  target_roles:
    - mafia
  default_config:
    vector_path: ${PERSONA_VECTORS_ROOT}/bf16/opportunistic.pt
    coefficient: 1.25
    layer_path_template: "model.layers.{}"
```

Config: `config/games/mafia/mafia_activation_wolf_opportunistic_8p.yaml`

At the start of each episode the console prints which seat was armed:
```
[RoleAwareSteering] Steering {'player_2'} (matched target roles {'mafia'})
```

## Persona probing

Passively observe which persona traits each agent's completions align with —
including agents that are NOT being steered — using the same trait vectors used
for steering.

### How it works

During `model.generate()` a read-only hook collects the final-token hidden
state at the probe layer (default: layer 20, same as the steering vectors).
It accumulates an exponential moving average over a configurable token window
(default 10 tokens), then at the end of each generation projects the mean
state onto every trait vector in `vectors_dir` via cosine similarity.

The result is a `persona_probe` dict `{trait: score}` logged to every JSONL
step record. No extra inference is required — the hook is free, adding only dot
products.

### Enabling probing

Add a `probing:` block to any config YAML:

```yaml
probing:
  enabled: true
  vectors_dir: ${PERSONA_VECTORS_ROOT}/bf16
  layer: 20
  layer_path_template: "model.layers.{}"
  window_tokens: 10   # EMA window (0 = cumulative mean over whole completion)
  top_k: 5            # traits reported in trace + wandb
```

The `.trace.txt` sidecar shows the top-5 probe scores per turn:
```
[PROBE top-5] opportunistic=0.412, charismatic=0.301, assertive=0.289, ...
```

### Analysing results

```bash
# Steered run vs. control comparison
python scripts/textarena/analyze_persona_probes.py \
    --steered  logs/mafia/mafia_activation_wolf_opportunistic_8p \
    --control  logs/mafia/mafia_noop_8p \
    --top-k 5 \
    --out reports/mafia_probe_analysis.txt
```

Example output:
```
── Effect on steered agent (player_2) ─────────────────────────────────────

  Trait                               Steered   Control       Δ
  ─────────────────────────────────── ────────  ────────  ───────  ─────
  opportunistic                        +0.412    +0.183   +0.229   ★
  charismatic                          +0.301    +0.289   +0.012
  trustworthiness                      +0.121    +0.151   -0.030

── Cross-agent spill-over (unsteered agents) ───────────────────────────────

  No significant trait shifts (|Δ| ≥ 0.04) detected in unsteered agents.
```

★ marks effects where |Δ| ≥ 0.05. Scores are cosine projections: positive
means the hidden state is aligned with that trait's learned direction.

### Adding a new trait

Drop `<trait>.pt` / `<trait>.json` into `$PERSONA_VECTORS_ROOT/bf16/` and
create a YAML pointing at it. No code changes required.

## Adding a new game

See **[docs/games/adding_a_new_game.md](docs/games/adding_a_new_game.md)** for the full step-by-step checklist.

1. Add `Adapter` + `Renderer` + `Parser` under `testbed/envs/symbolic/`
2. Register in `testbed/registry.py`
3. Add configs under `config/games/<game_id>/`
4. Write scripts under `scripts/textarena/`
5. Add `docs/games/<game_id>.md`

## Tests

```bash
python -m pytest -q
```
