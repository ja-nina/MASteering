# LLM Steering Multi-Game Testbed

A research testbed for measuring how **prompt injections** and **activation steering** change LLM agent behaviour across text-based multi-agent games.

## Overview

Agents powered by local LLMs play coordination games. Steering methods are applied per-agent and every turn is logged for behavioural analysis.

**Games supported**

| Family | ID | Players | Status | Doc |
|--------|----|---------|--------|-----|
| `symbolic` | `gbs` | 2–10 | Fully implemented | [gbs.md](docs/games/gbs.md) |
| `symbolic` | `gbs_exact_replication` | 2–10 | Fully implemented (Riedl 2025 replication) | [gbs.md](docs/games/gbs.md) |
| `symbolic` | `beauty_contest` | 2–10 | Fully implemented | [beauty_contest.md](docs/games/beauty_contest.md) |
| `symbolic` | `avalon` | 5–10 | Skeleton — parser+renderer done, submit() pending | [avalon.md](docs/games/avalon.md) |
| `symbolic` | `hanabi` | 2–5 | Skeleton — parser+renderer done, submit() pending | [hanabi.md](docs/games/hanabi.md) |
| `symbolic` | `werewolf` | 4–10 | Skeleton — parser+renderer done, submit() pending | [werewolf.md](docs/games/werewolf.md) |
| `symbolic` | `overcooked` | 2 | Skeleton — parser+renderer done, grid sim pending | [overcooked.md](docs/games/overcooked.md) |
| `textarena` | `Diplomacy-v0` | 5–7 | Fully playable via TextArena | [diplomacy.md](docs/games/diplomacy.md) |
| `textarena` | any TextArena `env_id` | varies | Fully playable — no game code needed | [textarena.md](docs/games/textarena.md) |

**Steering methods**

| Method | Description |
|--------|-------------|
| `noop` | Baseline — no steering applied |
| `prompt_injection` | Per-agent system suffix / user prefix injected at inference time |
| `activation` | Residual-stream vector addition at a chosen decoder layer |

## Quickstart

```bash
pip install -r requirements.txt

# Run one episode (GBS picking game, useful for debugging)
python scripts/run_episode.py \
  --config config/experiments/picking_sweep/gbs_exact_replication_plain_2p_14b.yaml \
  --episodes 1
```

Episode logs land in `logs/picking_sweep/<run_id>/episode_N.jsonl` with a `.summary.json` sidecar.

## Repository layout

```
testbed/                        # Core library — game-agnostic
  orchestrator.py               # Episode loop (parallel agent calls via ThreadPoolExecutor)
  registry.py                   # Maps game_id → (Adapter, Renderer, Parser)
  envs/
    symbolic/                   # Simultaneous-move games
      beauty_contest.py
      gbs.py
    textarena/
      ta_adapter.py             # Wraps the TextArena library
  renderers/symbolic/           # State → prompt text, one file per game
  parsers/symbolic/             # Text → action, one file per game
  steering/
    noop.py
    prompt_injection.py
    activation.py
  policy/
    transformers_policy.py      # In-process HuggingFace inference
    vllm_policy.py              # OpenAI-compatible vLLM client
  logging_/episode_logger.py

config/
  shared/                       # Persona lists shared across games
    personas.yaml               # 20 character personas (Riedl 2025)
    behavioral_personas.yaml    # 34 behavioral archetypes
    run_config.yaml             # Annotated config template
  games/                        # Individual hand-written configs, one directory per game
    gbs/
    beauty_contest/
    textarena/
  experiments/                  # Sweep configs (may span multiple games)
    picking_sweep/
    exact_replication_sweep/
    persona_sweep/
    layer_sweep/
    reasoning_sweep/

cases/
  gbs/
    persona_impact_cases.json   # 50 extracted GBS game states for counterfactual study

scripts/
  run_episode.py                # Game-agnostic episode runner (primary CLI)
  analyze_results.py            # CSV aggregation across run_ids
  gbs/                          # GBS-specific experiment scripts
    picking_sweep/
      gen_picking_configs.py
      plot_picking_sweep.py
    persona_sweep/
      gen_persona_sweep_configs.py
      plot_persona_sweep.py
    persona_impact/
      extract_persona_cases.py
      run_persona_cases.py
      merge_persona_cases.py
      plot_persona_impact.py
    steering/
      collect_activations.py
      extract_steering_vector.py
      train_sae.py
      find_tom_features.py
      run_layer_sweep.py
    slurm/                      # SLURM array job scripts (cluster-specific paths)
  experiments/                  # Multi-game experiment scripts
    reasoning_sweep/
      gen_reasoning_sweep_configs.py
      plot_reasoning_sweep.py
  beauty_contest/               # Placeholder for future BC-specific scripts

docs/games/                     # Per-game documentation
  gbs.md
  beauty_contest.md
  textarena.md
  adding_a_new_game.md          # Step-by-step guide

logs/                           # Runtime outputs (gitignored)
plots/                          # Analysis figures
figures/                        # Publication figures
```

## Adding a new game

See **[docs/games/adding_a_new_game.md](docs/games/adding_a_new_game.md)** for the full step-by-step checklist.

The short version:
1. Add `Adapter` + `Renderer` + `Parser` under `testbed/envs/symbolic/` (or use `textarena` family)
2. Register in `testbed/registry.py`
3. Add configs under `config/games/<game_id>/`
4. Write scripts under `scripts/<game_id>/`
5. Add `docs/games/<game_id>.md`

## GBS picking sweep

Systematic sweep over steering conditions (plain / persona / Theory-of-Mind) × player counts (2 / 3 / 10):

```bash
# Generate configs
python scripts/gbs/picking_sweep/gen_picking_configs.py

# Run on cluster
sbatch scripts/gbs/slurm/run_exact_gbs_replication_qwen.slurm
sbatch scripts/gbs/slurm/run_exact_gbs_replication_20b.slurm

# Plot
python scripts/gbs/picking_sweep/plot_picking_sweep.py
```

Outputs: `figures/picking_sweep/{qwen3_14b,gpt_oss_20b}/*.png`

## GBS persona impact study

50 game-state cases × 70 conditions × 20 reps = 70,000 inferences:

```bash
python scripts/gbs/persona_impact/extract_persona_cases.py
sbatch scripts/gbs/slurm/run_persona_cases.slurm
python scripts/gbs/persona_impact/merge_persona_cases.py
python scripts/gbs/persona_impact/plot_persona_impact.py
```

## Tests

```bash
python -m pytest -q
```
