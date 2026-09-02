#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/measure_kl_%j.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/measure_kl_%j.err
#SBATCH -t 4:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  KL calibration sweep — gpu22 variant
#
#  Identical to run_measure_kl.slurm but targets the gpu22 partition.
#
#  Launch:
#    sbatch scripts/training/slurm/run_measure_kl_gpu22.slurm
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LORA_ROOT="/scratch/inf0/user/nzukowsk/MASteer/loras_ipd_nudge_v3_nopersona"
OUT_FILE="/scratch/inf0/user/nzukowsk/MASteer/kl_calibration.json"

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

cd "${REPO}"

echo "[$(date)] Job ${SLURM_JOB_ID}"
echo "[$(date)] GPU : $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "[$(date)] Glob: ${LORA_ROOT}/ipdNV3np_*"
echo "[$(date)] Out : ${OUT_FILE}"

"$PYTHON" -u "${REPO}/scripts/measure_kl.py" \
    --model              Qwen/Qwen3-4B                        \
    --lora-glob          "${LORA_ROOT}/ipdNV3np_*"            \
    --n-samples          20                                    \
    --max-new-tokens     200                                   \
    --temperature        0.7                                   \
    --device             cuda                                  \
    --dtype              bfloat16                              \
    --output             "${OUT_FILE}"

echo "[$(date)] Done — results at ${OUT_FILE}"
