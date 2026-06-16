#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=10-35%8        # layers 10-35 (Qwen3-4B has 36 layers, 0-35); %8 = 8 concurrent tasks
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/layer_sweep_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/layer_sweep_%A_%a.err
#SBATCH -t 6:00:00
#
# Job-array sweep: one task per layer, up to 8 running in parallel.
# Each task: collect activations → train SAE (resid + MLP) → find ToM vector → delete large file.
# Peak disk: 8 × ~90 GB ≈ 720 GB, well within 1 TB scratch budget.
#
# Submit once per game from the repo root:
#
#   sbatch scripts/run_layer_sweep.slurm beauty_contest
#   sbatch scripts/run_layer_sweep.slurm gbs
#
# Override array range at submission time, e.g. to resume from layer 20:
#   sbatch --array=20-35%8 scripts/run_layer_sweep.slurm beauty_contest
#
# Time a single layer before setting -t above:
#   sbatch --array=10 scripts/run_layer_sweep.slurm beauty_contest

set -euo pipefail
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME}"

GAME="${1:-beauty_contest}"               # beauty_contest | gbs

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

"$PYTHON" scripts/run_layer_sweep.py \
  --game "${GAME}" \
  --model Qwen/Qwen3-4B \
  --layer-index "${SLURM_ARRAY_TASK_ID}" \
  --num-episodes 500 \
  --max-rounds 15 \
  --num-players 4 \
  --activations-dir /scratch/inf0/user/nzukowsk/activations \
  --sae-dir /scratch/inf0/user/nzukowsk/sae \
  --vectors-dir vectors \
  --wandb

echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} finished."
