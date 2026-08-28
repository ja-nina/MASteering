#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/probe_only_%j.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/probe_only_%j.err
#SBATCH -t 0:30:00
#
# ════════════════════════════════════════════════════════════════════════════
#  Probe-only smoke test  (no judge, no vLLM needed)
#
#  Validates that:
#    1. Model loads and generates steered rollouts
#    2. probe_all captures cosine-sims at every layer for every trait
#    3. Output JSONL format is correct
#
#  Prints a 300-char response snippet + top-5/bot-5 probe scores per record.
#  No judge API call — safe to run on a single GPU in <10 minutes.
#
#  Inspect output:
#    cat /scratch/inf0/user/nzukowsk/MASteer/test_probe_only/results.jsonl \
#        | python -m json.tool | head -100
#
#  Launch:
#    sbatch scripts/training/slurm/run_judge_calibration_probe_only.slurm
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
RAW_VECTORS_DIR="/BS/ForNet/work/MASteering/data/vector_extraction/persona/qwen3-4b/bf16"
OUT_DIR="/scratch/inf0/user/nzukowsk/MASteer/test_probe_only"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs "${OUT_DIR}"

echo "[$(date)] Job ${SLURM_JOB_ID} — probe-only smoke test (no judge)"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'n/a')"

"$PYTHON" -u "${REPO}/scripts/eval_judge_calibration.py" \
    --model           "Qwen/Qwen3-4B"       \
    --raw-vectors-dir "${RAW_VECTORS_DIR}"  \
    --traits          "evil"                \
    --steer-layer     14                    \
    --steer-alpha     1.5                   \
    --n-rollouts      2                     \
    --max-prompts     3                     \
    --probe-all-traits                      \
    --skip-judge                            \
    --verbose                               \
    --output-dir      "${OUT_DIR}"          \
    --device          cuda:0

echo "[$(date)] Done. Check ${OUT_DIR}/results.jsonl"
