#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0      # task 0 = baselines; tasks 10-35 = per-layer activation
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/evals_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/evals_%A_%a.err
#SBATCH -t 4:00:00
#
# Task 0  — noop + prompt injection baselines (run once, no layer dependency)
# Tasks 10-35 — activation steering for each layer (requires layer sweep + gen_layer_configs.py)
#
# Prerequisites:
#   Layer sweep:   sbatch scripts/run_layer_sweep.slurm <game>
#   Config gen:    python scripts/gen_layer_configs.py
#
# Submit:
#   sbatch scripts/run_evals.slurm
#
# Baselines only:
#   sbatch --array=0 scripts/run_evals.slurm
#
# One activation layer manually:
#   sbatch --array=18 scripts/run_evals.slurm

set -euo pipefail
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME}"

L="${SLURM_ARRAY_TASK_ID}"

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

if [ "${L}" -eq 0 ]; then
  # ── baselines: noop + prompt injection ──────────────────────────────────────
  CONFIGS=(
    "config/exp_noop.yaml"
    "config/exp_prompt_one.yaml"
    "config/exp_prompt_all.yaml"
    "config/gbs_exp_noop.yaml"
    "config/gbs_exp_noop_directional.yaml"
    "config/gbs_exp_prompt_one.yaml"
    "config/gbs_exp_prompt_all.yaml"
    "config/gbs_exp_prompt_all_directional.yaml"
  )
else
  # ── activation steering for layer L ─────────────────────────────────────────
  CONFIGS=(
    "config/layer_sweep/exp_activation_all_l${L}.yaml"
    "config/layer_sweep/exp_activation_one_l${L}.yaml"
    "config/layer_sweep/gbs_exp_activation_all_l${L}.yaml"
    "config/layer_sweep/gbs_exp_activation_one_l${L}.yaml"
  )
fi

for CONFIG in "${CONFIGS[@]}"; do
  echo "[$(date)] Running ${CONFIG}"
  "$PYTHON" scripts/run_episode.py --config "${CONFIG}"
  echo "[$(date)] Done     ${CONFIG}"
done

echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} finished."
