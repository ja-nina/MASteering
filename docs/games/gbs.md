# Goldstone Group Sum (GBS)

**Family:** `symbolic` | **IDs:** `gbs`, `gbs_exact_replication`

---

## Rules

N players simultaneously submit an integer contribution. Their contributions are
summed and compared to a hidden target T. Agents receive feedback after each round
and adjust in the next round. The episode ends when the group sum equals T, or after
`num_rounds` rounds.

**Target range (default):** `T ~ Uniform(5N, 50N)`, seeded per episode.

**Reward:** 1.0 on the round the group succeeds, 0.0 otherwise.

---

## Action format

**Standard GBS** (`hide_group_size=False`):
> Respond with `NUMBER: <integer>` where `<integer>` ≥ 0.

**Picking / Exact-replication** (`hide_group_size=True`):
> Respond with `FINAL GUESS: <integer>` where `<integer>` ∈ [0, 50].

Agents do not know N when `hide_group_size=True`; they only see their own
contribution history and the group-level feedback.

---

## Feedback modes

| `feedback` value | What the agent sees |
|---|---|
| `"directional"` | `"too HIGH"` / `"too LOW"` / `"correct"` |
| `"exact"` | Signed error: `group_sum - target` |

---

## Persona system

GBS supports two persona configs (both in `config/shared/`):

| File | Contents | Used in |
|---|---|---|
| `personas.yaml` | 20 character personas (Riedl 2025 replication) | picking sweep |
| `behavioral_personas.yaml` | 34 behavioral archetypes | persona impact study |

Set via `env_kwargs` in the YAML config:
```yaml
env_kwargs:
  personas: [...]          # list of persona strings; adapter samples without replacement
  persona_mode: persona    # "plain" | "persona" | "tom"
```

`persona_mode: tom` appends the Theory-of-Mind instruction to the system prompt.

---

## Config keys (`env_kwargs`)

| Key | Default | Description |
|---|---|---|
| `num_rounds` | 20 | Maximum rounds before episode ends |
| `low` | 0 | Minimum valid contribution |
| `high` | — | Maximum valid contribution (defaults to 50 × N) |
| `target` | random | Fixed target; if omitted, drawn from Uniform(5N, 50N) |
| `feedback` | `"exact"` | `"directional"` or `"exact"` |
| `hide_group_size` | `False` | Use `FINAL GUESS:` keyword; agents don't know N |
| `personas` | `[]` | List of persona strings to assign without replacement |
| `persona_mode` | `"plain"` | `"plain"`, `"persona"`, or `"tom"` |
| `seed_base` | — | If set, overrides per-episode seed for reproducible targets |

---

## Game aliases

`gbs_exact_replication` is registered as an alias for `GBSAdapter` with
`hide_group_size=True` and `feedback="directional"` expected in `env_kwargs`.
It faithfully replicates Riedl (2025, arXiv 2510.05174).

---

## Experiments

### Picking sweep (`config/experiments/picking_sweep/`)

Systematic sweep: plain / persona / ToM × 2p / 3p / 10p × Qwen3-14B / gpt-oss-20b.
200 episodes per cell.

```bash
python scripts/gbs/picking_sweep/gen_picking_configs.py
sbatch scripts/gbs/slurm/run_exact_gbs_replication_qwen.slurm
python scripts/gbs/picking_sweep/plot_picking_sweep.py
```

Outputs: `figures/gbs/picking_sweep/` (or `figures/picking_sweep/` for legacy runs).

### Persona sweep (`config/experiments/persona_sweep/`)

21 conditions: plain + 20 behavioral archetypes. 2 players, 100 episodes.

```bash
python scripts/gbs/persona_sweep/gen_persona_sweep_configs.py
sbatch scripts/gbs/slurm/run_persona_sweep_qwen.slurm
python scripts/gbs/persona_sweep/plot_persona_sweep.py
```

### Persona impact / counterfactual study

50 extracted game states (cases) × 70 conditions × 20 reps = 70,000 inferences.
Tests whether personas shift agent behaviour on fixed game situations.

```bash
# Extract cases from existing plain logs
python scripts/gbs/persona_impact/extract_persona_cases.py

# Run counterfactual inferences (SLURM array: 630 tasks)
sbatch scripts/gbs/slurm/run_persona_cases.slurm

# Merge per-job files
python scripts/gbs/persona_impact/merge_persona_cases.py

# Plot (15 figures)
python scripts/gbs/persona_impact/plot_persona_impact.py
```

Cases stored in: `cases/gbs/persona_impact_cases.json`  
Results in: `logs/persona_impact/case_results.jsonl`  
Figures in: `plots/persona_impact/`

### Layer sweep (`config/experiments/layer_sweep/`)

Sweeps residual-stream layers 10–35 of Qwen3-4B through the activation
steering pipeline: collect → train SAE → find ToM features → save vector.

```bash
sbatch --array=10-35%8 scripts/gbs/slurm/run_layer_sweep.slurm gbs
```

### Activation steering

Direct CAA (no SAE):

```bash
python scripts/gbs/steering/extract_steering_vector.py \
    --model Qwen/Qwen3-4B --layer model.layers.18 \
    --game gbs --num-samples 64 \
    --output vectors/tom_gbs_l18.npy
```

Via SAE:

```bash
python scripts/gbs/steering/collect_activations.py --game gbs --layer model.layers.18 ...
python scripts/gbs/steering/train_sae.py --activations activations/base_gbs_... ...
python scripts/gbs/steering/find_tom_features.py --sae saes/... --top-n 16 ...
```

---

## Implementation files

| Purpose | File |
|---|---|
| Game logic | `testbed/envs/symbolic/gbs.py` |
| Prompt renderer | `testbed/renderers/symbolic/gbs.py` |
| Action parser | `testbed/parsers/symbolic/gbs.py` |
| Registry entry | `testbed/registry.py` |
