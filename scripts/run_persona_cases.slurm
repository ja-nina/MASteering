#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-629%30
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/persona_logs/persona_cases_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/persona_logs/persona_cases_%A_%a.err
#SBATCH -t 1:55:00
#
# Persona-impact counterfactual experiment.
#
# 42 conditions x 9 case-chunks = 378 array tasks.
# Each task handles 6 cases x 20 reps = 120 model calls (≤ 130 limit).
#
# Array layout:
#   task_id = condition_idx * N_CHUNKS + chunk_idx
#   condition_idx in [0, 41]  (plain, plain_tom, cooperative, cooperative_tom, ...)
#   chunk_idx     in [0,  8]  (each covers 6 of the 50 cases)
#
# Submit:
#   sbatch scripts/run_persona_cases.slurm

set -euo pipefail

N_CONDITIONS=70
N_CHUNKS=9          # ceil(50 cases / 6 cases-per-chunk)
CASES_PER_CHUNK=6

T=${SLURM_ARRAY_TASK_ID}
CONDITION_IDX=$((T / N_CHUNKS))
CHUNK=$((T % N_CHUNKS))

echo "[$(date)] Job ${SLURM_JOB_ID} task ${T} | condition_idx=${CONDITION_IDX} chunk=${CHUNK} on ${HOSTNAME}"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

"$PYTHON" scripts/run_persona_cases.py \
    --condition-idx "${CONDITION_IDX}" \
    --chunk         "${CHUNK}" \
    --cases-per-chunk "${CASES_PER_CHUNK}"

echo "[$(date)] Task ${T} done."
