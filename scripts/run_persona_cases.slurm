#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-377%30
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/persona_cases_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/persona_cases_%A_%a.err
#SBATCH -t 1:30:00
#
# Persona-impact counterfactual experiment.
#
# 42 conditions x 9 case-chunks = 378 array tasks.
# Each task handles 6 cases x 20 reps = 120 model calls (≤ 130 limit).
#
# Submit:
#   sbatch scripts/run_persona_cases.slurm

set -euo pipefail

N_CHUNKS=9
CASES_PER_CHUNK=6
REPS=20
RESUME_HOP_LIMIT=20

HOP=${RESUME_HOP:-0}
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME} (hop ${HOP})"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

T=${SLURM_ARRAY_TASK_ID}
CONDITION_IDX=$((T / N_CHUNKS))
CHUNK=$((T % N_CHUNKS))

echo "[$(date)] condition_idx=${CONDITION_IDX} chunk=${CHUNK}"

"$PYTHON" scripts/run_persona_cases.py \
    --condition-idx "${CONDITION_IDX}" \
    --chunk         "${CHUNK}" \
    --cases-per-chunk "${CASES_PER_CHUNK}"

echo "[$(date)] Job ${SLURM_JOB_ID} task ${T} finished this run."

# Check how many records still missing in the per-job output file
JOB_OUT="logs/persona_impact/results_cond$(printf '%02d' ${CONDITION_IDX})_chunk$(printf '%02d' ${CHUNK}).jsonl"

REMAINING=$("$PYTHON" - "${JOB_OUT}" "${CHUNK}" "${CASES_PER_CHUNK}" "${REPS}" <<'PYEOF'
import json, sys
from pathlib import Path

job_out      = Path(sys.argv[1])
chunk        = int(sys.argv[2])
per_chunk    = int(sys.argv[3])
reps         = int(sys.argv[4])

cases      = json.loads(Path("cases/persona_impact_cases.json").read_text(encoding="utf-8"))
start      = chunk * per_chunk
n_cases    = min(per_chunk, max(0, len(cases) - start))
expected   = n_cases * reps

done = 0
if job_out.exists():
    for line in job_out.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done += 1

print(max(0, expected - done))
PYEOF
)

if [ "${REMAINING}" -gt 0 ] && [ "${HOP}" -lt "${RESUME_HOP_LIMIT}" ]; then
    echo "[$(date)] ${REMAINING} inference(s) remaining for task ${T} — requeuing (hop $((HOP + 1)))"
    sbatch --dependency=afterany:"${SLURM_JOB_ID}" --array="${T}" \
           --export=ALL,RESUME_HOP=$((HOP + 1)) "$0"
elif [ "${REMAINING}" -gt 0 ]; then
    echo "[$(date)] WARNING: hop limit (${RESUME_HOP_LIMIT}) reached with ${REMAINING} remaining — not requeuing."
else
    echo "[$(date)] All inferences complete for task ${T}."
fi
