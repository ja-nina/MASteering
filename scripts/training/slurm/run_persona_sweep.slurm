#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --array=0-23
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/persona_sweep_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/persona_sweep_%A_%a.err
#SBATCH -t 24:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  Persona sweep — one job per target trait combination.
#  Each array task trains both adapters (SVD + free LoRA) toward a
#  different personality direction in the residual-stream SVD basis.
#
#  Launch:
#    sbatch scripts/training/slurm/run_persona_sweep.slurm
#
#  To run a subset:
#    sbatch --array=0-3 scripts/training/slurm/run_persona_sweep.slurm
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
CONFIG="${REPO}/configs/training/persona_lora.yaml"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export SVD_BASIS="${REPO}/data/svd_basis/qwen3-4b-residual.pt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

# ── Persona definitions ──────────────────────────────────────────────────────
# Format: "run_name|trait:weight"  — one trait per job
PERSONAS=(
    # maximize (IDs 0-11)
    "max_agreeable|agreeable:1.0"
    "max_warm|warm:1.0"
    "max_dominant|dominant:1.0"
    "max_confident|confident:1.0"
    "max_curious|curious:1.0"
    "max_empathetic|empathetic:1.0"
    "max_hostile|hostile:1.0"
    "max_anxious|anxious:1.0"
    "max_creative|creative:1.0"
    "max_assertive|assertive:1.0"
    "max_cold|cold:1.0"
    "max_neurotic|neurotic:1.0"
    # minimize (IDs 12-23) — negative weight flips the target direction
    "min_agreeable|agreeable:-1.0"
    "min_warm|warm:-1.0"
    "min_dominant|dominant:-1.0"
    "min_confident|confident:-1.0"
    "min_curious|curious:-1.0"
    "min_empathetic|empathetic:-1.0"
    "min_hostile|hostile:-1.0"
    "min_anxious|anxious:-1.0"
    "min_creative|creative:-1.0"
    "min_assertive|assertive:-1.0"
    "min_cold|cold:-1.0"
    "min_neurotic|neurotic:-1.0"
)

ENTRY="${PERSONAS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="${ENTRY%%|*}"
TRAITS="${ENTRY##*|}"

cd "${REPO}"

echo "[$(date)] Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Persona : ${RUN_NAME}"
echo "[$(date)] Traits  : ${TRAITS}"
echo "[$(date)] GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"

"$PYTHON" -u "${REPO}/scripts/train_lora.py" \
    --config   "${CONFIG}"  \
    --traits   "${TRAITS}"  \
    --run-name "${RUN_NAME}"

echo "[$(date)] Done — ${RUN_NAME}"
