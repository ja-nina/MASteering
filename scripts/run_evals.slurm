#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=10-35%8        # one task per layer; %8 = 8 concurrent evals
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/evals_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/evals_%A_%a.err
#SBATCH -t 4:00:00
#
# Runs all four activation-steering eval configs for layer $SLURM_ARRAY_TASK_ID:
#   beauty_contest / all agents
#   beauty_contest / player_0 only
#   gbs            / all agents
#   gbs            / player_0 only
#
# Prerequisites:
#   1. Layer sweep complete  (scripts/run_layer_sweep.slurm)
#   2. Configs generated     (python scripts/gen_layer_configs.py)
#
# Submit:
#   sbatch scripts/run_evals.slurm
#
# To run a single layer manually:
#   sbatch --array=18 scripts/run_evals.slurm

set -euo pipefail
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} (layer ${SLURM_ARRAY_TASK_ID}) starting on ${HOSTNAME}"

L="${SLURM_ARRAY_TASK_ID}"

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

CONFIGS=(
  "config/layer_sweep/exp_activation_all_l${L}.yaml"
  "config/layer_sweep/exp_activation_one_l${L}.yaml"
  "config/layer_sweep/gbs_exp_activation_all_l${L}.yaml"
  "config/layer_sweep/gbs_exp_activation_one_l${L}.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
  echo "[$(date)] Running ${CONFIG}"
  "$PYTHON" scripts/run_episode.py --config "${CONFIG}"
  echo "[$(date)] Done     ${CONFIG}"
done

echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} finished."
