#!/bin/bash
#SBATCH --job-name=ma-games
#SBATCH -p gpu-a40
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-logs/games-%A_%a.out
# Usage: sbatch [--array=0-N] scripts/slurm/run_games.sbatch <config dir> [run_games.py args]
# All configs under <config dir> share one steer + one monitor engine (2 GPUs, see engine: in the
# configs). As an array, task i plays shard i of the episodes. Resubmitting resumes.
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/env.sh"
CONFIG_DIR="${1:?Usage: sbatch scripts/slurm/run_games.sbatch <config dir> [args]}"; shift
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
  set -- "$@" --shard "$((SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN))" --num-shards "$SLURM_ARRAY_TASK_COUNT"
fi
"$PYTHON" scripts/run_games.py --config-dir "$CONFIG_DIR" "$@"
