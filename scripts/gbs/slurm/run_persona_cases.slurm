#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-629%15
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/persona_logs/persona_cases_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/persona_logs/persona_cases_%A_%a.err
#SBATCH -t 1:55:00
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07

# Persona-impact counterfactual experiment.
#
# 70 conditions x 9 case-chunks = 630 array tasks.
# Each task handles up to 6 cases x 20 reps = 120 model calls.
#
# Array layout:
#   task_id = condition_idx * N_CHUNKS + chunk_idx
#   condition_idx in [0, 69]  (plain, plain_tom, 34 personas x 2)
#   chunk_idx     in [0,  8]  (each covers up to 6 of the 50 cases)
#
# Submit:
#   sbatch scripts/run_persona_cases.slurm

set -uo pipefail

N_CONDITIONS=70
N_CHUNKS=9
CASES_PER_CHUNK=6
RESUME_HOP_LIMIT=10

HOP=${RESUME_HOP:-0}
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME} (hop ${HOP})"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/persona_logs

T=${SLURM_ARRAY_TASK_ID}
CONDITION_IDX=$((T / N_CHUNKS))
CHUNK=$((T % N_CHUNKS))

echo "[$(date)] condition_idx=${CONDITION_IDX} chunk=${CHUNK}"

_requeue() {
  JOB_FILE="logs/persona_impact/results_cond$(printf '%02d' ${CONDITION_IDX})_chunk$(printf '%02d' ${CHUNK}).jsonl"
  if [ -f "${JOB_FILE}" ]; then
    REMAINING=0
  else
    REMAINING=1
  fi

  if [ "${REMAINING}" -gt 0 ] && [ "${HOP}" -lt "${RESUME_HOP_LIMIT}" ]; then
    echo "[$(date)] chunk not complete — requeuing (hop $((HOP + 1)))"
    sbatch --dependency=afterany:"${SLURM_JOB_ID}" --array="${T}" \
           --export=ALL,RESUME_HOP=$((HOP + 1)) "$0"
  elif [ "${REMAINING}" -gt 0 ]; then
    echo "[$(date)] WARNING: hop limit (${RESUME_HOP_LIMIT}) reached, chunk still incomplete."
  else
    echo "[$(date)] Chunk complete (cond${CONDITION_IDX} chunk${CHUNK})."
  fi
}

_on_term() {
  echo "[$(date)] SIGTERM received — stopping Python and requeuing."
  kill -TERM "${PYTHON_PID}" 2>/dev/null || true
  wait "${PYTHON_PID}" || true
  _requeue
  exit 0
}
trap '_on_term' TERM

"$PYTHON" scripts/gbs/persona_impact/run_persona_cases.py \
    --condition-idx "${CONDITION_IDX}" \
    --chunk         "${CHUNK}" \
    --cases-per-chunk "${CASES_PER_CHUNK}" &
PYTHON_PID=$!
wait "${PYTHON_PID}"
PYTHON_EXIT=$?

echo "[$(date)] Python finished (exit ${PYTHON_EXIT})."
_requeue
