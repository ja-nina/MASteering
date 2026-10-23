#!/bin/bash
#SBATCH --job-name=ma-monitor
#SBATCH -p gpu-a40
#SBATCH -c 16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/monitor-%A_%a.out
# Usage: sbatch [--array=0-N] scripts/slurm/backfill_monitor.sbatch <config dir> [backfill_monitor.py args]
# Monitor scores of the episodes a game group played without the monitor (scripts/backfill_monitor.py): the
# monitor engine alone, reading the logged turns. As an array, task i reads shard i. Resubmitting resumes.
# 27B on 4 a40s (the monitor's buffer counts twice: see config/sweeps/focus.yaml):
#   sbatch --array=0-9 scripts/slurm/backfill_monitor.sbatch config/runs/focus_qwen3.8-27b/gbs_6p --engine \
#     '{"aperture_gib": 8, "capture_tokens": 3072, "monitor": {"gpus": "0,1,2,3", "tensor_parallel_size": 4,
#       "memory": 0.8, "max_num_batched_tokens": 4096, "disable_custom_all_reduce": true}}'
# Check first against episodes with live scores (9B, one GPU):
#   sbatch --gres=gpu:1 scripts/slurm/backfill_monitor.sbatch config/runs/focus/gbs_3p --verify 10 \
#     --engine '{"monitor": {"gpus": "0", "memory": 0.75}}'
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/env.sh"
CONFIG_DIR="${1:?Usage: sbatch scripts/slurm/backfill_monitor.sbatch <config dir> [args]}"; shift
if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]; then
  case " $* " in *" --num-shards "*) ;; *) set -- "$@" --num-shards "$SLURM_ARRAY_TASK_COUNT" ;; esac
  set -- "$@" --shard "${SLURM_ARRAY_TASK_ID}"
fi
"$PYTHON" scripts/backfill_monitor.py --config-dir "$CONFIG_DIR" "$@"
