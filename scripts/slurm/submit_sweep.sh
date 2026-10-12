#!/bin/bash
# Submit one SLURM array per game group of a generated sweep.
# Usage: bash scripts/slurm/submit_sweep.sh config/runs/main [shards per group, default 8] [extra sbatch args]
set -euo pipefail
SWEEP="${1:?Usage: submit_sweep.sh <config/runs/NAME> [shards] [sbatch args]}"
SHARDS="${2:-8}"; shift; shift || true
for GROUP in "$SWEEP"/*/; do
  echo "$(basename "$GROUP"): $(sbatch --parsable --array=0-$((SHARDS - 1)) "$@" scripts/slurm/run_games.sbatch "$GROUP")"
done
