#!/bin/bash
#SBATCH --job-name=ma-games
#SBATCH -p gpu-a40
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=7:00:00
#SBATCH --output=slurm-logs/games-%A_%a.out
# Usage: sbatch [--array=0-N] scripts/slurm/run_games.sbatch <config dir> [run_games.py args]
# All configs under <config dir> share one steer + one monitor engine (2 GPUs, see engine: in the
# configs). As an array, task i plays shard i of the episodes. Resubmitting resumes.
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/env.sh"
CONFIG_DIR="${1:?Usage: sbatch scripts/slurm/run_games.sbatch <config dir> [args]}"; shift
# task i plays shard i of --num-shards (default: the array's size). To rerun single tasks of a 29-task array:
#   sbatch --array=21,23 scripts/slurm/run_games.sbatch <config dir> --num-shards 29
# (not via sbatch --export=...,VAR=...: on this cluster such jobs end "user env retrieval failed requeued held")
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
  case " $* " in *" --num-shards "*) ;; *) set -- "$@" --num-shards "$SLURM_ARRAY_TASK_COUNT" ;; esac
  set -- "$@" --shard "${SLURM_ARRAY_TASK_ID}"
fi
"$PYTHON" scripts/run_games.py --config-dir "$CONFIG_DIR" "$@"
