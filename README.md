# LLM Steering Multi-Game Testbed

A research testbed for measuring how **prompt injections** change LLM agent behaviour across text-based multi-agent games.

## Overview

Agents powered by local LLMs play coordination games. Steering methods are applied per-agent and every turn is logged for behavioural analysis.

**Games supported**

| Family | ID | Description |
|--------|----|-------------|
| `symbolic` | `gbs` | Goldstone Group Sum — players each submit a contribution; the group wins when contributions sum to a hidden target |
| `symbolic` | `gbs_exact_replication` | Picking variant — `hide_group_size=True`, `feedback=directional` — faithfully replicates Riedl (2025, arXiv 2510.05174) |
| `symbolic` | `beauty_contest` | N players guess an integer; winner is closest to 2/3 of the group average |

**Steering methods**

| Method | Description |
|--------|-------------|
| `noop` | Baseline — no steering applied |
| `prompt_injection` | Per-agent system suffix / user prefix injected at inference time |

## Quickstart

```bash
pip install -r requirements.txt

# run one episode (single config, useful for debugging)
python scripts/run_episode.py \
  --config config/picking_sweep/gbs_exact_replication_plain_2p_14b.yaml \
  --episodes 1
```

Episode logs land in `logs/picking_sweep/<run_id>/episode_N.jsonl` with a `.summary.json` sidecar.

## Picking sweep (GBS exact-replication)

Systematic sweep over steering conditions (plain / persona / Theory-of-Mind) × player counts (2 / 3 / 10) using the `gbs_exact_replication` game alias (`hide_group_size=True`, `feedback=directional`). Target: **200 episodes per condition × player-count cell**.

### 1 — Generate configs

```bash
python scripts/gen_picking_configs.py
```

Writes one YAML per condition × player-count × model to `config/picking_sweep/`.

### 2 — Run on the cluster

```bash
sbatch scripts/run_exact_gbs_replication_qwen.slurm   # Qwen3-14B
sbatch scripts/run_exact_gbs_replication_20b.slurm    # gpt-oss-20b
```

Both scripts are self-requeueing array jobs (54 tasks = 3 conditions × 3 player counts × 6 shards). Each task:
- skips already-completed episodes via `.summary.json` detection
- re-submits itself with `--dependency=afterany` if episodes remain at walltime
- traps `SIGTERM` (walltime kill) so the requeue fires even when the job is killed mid-episode

Walltime: **23 h** (adjust `#SBATCH -t` if your cluster caps lower).

**Notes:**
- gpt-oss-20b on H100 nodes requires bfloat16. `TransformersPolicy` auto-detects this via `torch.cuda.get_device_name()`.
- A40 nodes (`gpu22-a40-05`, `gpu22-a40-06`) are excluded — MXFP4 kernels do not support Ampere.

### 3 — Plot results

```bash
python scripts/plot_picking_sweep.py
```

Reads every `*.summary.json` under `logs/picking_sweep/` and writes separate figure sets per model:

```
figures/picking_sweep/
  qwen3_14b/
    n_datapoints.png       — converged / non-converged / missing vs 200-episode target (stacked bars)
    convergence_line.png   — convergence rate vs group size (line chart, 95% Wilson CI)
    success_rate.png       — convergence rate per condition × player count (bars)
    rounds_to_success.png  — mean rounds, non-converged capped at 30 (bars)
    box_10p.png            — round distributions at 10 players (box plots, cap=30)
    violin.png             — full distributions all group sizes, median + mean marked
  gpt_oss_20b/
    (same set; cells with N < 10 are skipped)
```

Also writes `picking_sweep_summary.csv` (one row per episode).

## Project structure

```
testbed/
  orchestrator.py           # game-agnostic episode loop (parallel agent calls via ThreadPoolExecutor)
  envs/symbolic/gbs.py      # GBSAdapter + gbs_exact_replication alias
  renderers/symbolic/gbs.py # state → prompt text
  parsers/symbolic/gbs.py   # text → action, with error feedback
  steering/
    noop.py                 # NoOpSteering
    prompt_injection.py     # PromptInjectionSteering
  policy/
    transformers_policy.py  # in-process HF inference
    vllm_policy.py          # OpenAI-compatible vLLM client
  logging_/episode_logger.py
config/picking_sweep/       # one YAML per condition × player-count × model
scripts/
  gen_picking_configs.py    # generate config/picking_sweep/ YAMLs
  run_episode.py            # episode runner CLI
  plot_picking_sweep.py     # analysis figures
  run_exact_gbs_replication_qwen.slurm
  run_exact_gbs_replication_20b.slurm
logs/picking_sweep/         # episode JSONL + summary files (gitignored)
figures/picking_sweep/      # PNG outputs
picking_sweep_summary.csv   # flat per-episode summary
```

## Tests

```bash
python -m pytest -q
```
