#!/bin/bash
# Submit the cluster smoke tests: the vLLM-Hook engine checks and the all-games smoke sweep
# (they run in parallel, each on 2 GPUs). Run from the repo root with the pvf env active.
# Usage: bash scripts/slurm/submit_smoke.sh [model, default qwen3.5-9b] [extra sbatch args]
set -euo pipefail
MODEL="${1:-qwen3.5-9b}"; shift || true
ENGINES=$(sbatch --parsable "$@" scripts/slurm/smoke.sbatch "$MODEL")
GAMES=$(sbatch --parsable "$@" scripts/slurm/smoke_games.sbatch config/sweeps/smoke.yaml "$MODEL")
NAME=$(python scripts/make_configs.py config/sweeps/smoke.yaml --model "$MODEL" --print-name)
echo "engine checks: job $ENGINES -> slurm-logs/smoke-engines-$ENGINES.out, results/smoke/engines/$MODEL/"
echo "games smoke:   job $GAMES -> slurm-logs/smoke-games-$GAMES.out, results/$NAME/samples.html"
