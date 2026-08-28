#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 16
#SBATCH --gres=gpu:3
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/judge_calib_smoke_%j.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/judge_calib_smoke_%j.err
#SBATCH -t 2:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  Judge Calibration — SMOKE TEST
#
#  Validates the full pipeline end-to-end on a tiny subset before committing
#  to the full sweep.
#
#  GPU layout:
#    cuda:0        — Qwen3-4B (steered model, generation + probing)
#    cuda:1,2      — Qwen3-14B (vLLM judge server, tensor-parallel=2)
#
#  What it runs:
#    3 prompts × 2 rollouts × 1 steer_trait (evil) = 6 records
#    Each record: full probe_all (all traits × all layers) + blind judge_all
#
#  Inspect results:
#    cat /scratch/inf0/user/nzukowsk/MASteer/test_calib/results.jsonl | python -m json.tool | head -200
#
#  Launch:
#    sbatch scripts/training/slurm/run_judge_calibration_smoke.slurm
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
RAW_VECTORS_DIR="/BS/ForNet/work/MASteering/data/vector_extraction/persona/qwen3-4b/bf16"
JUDGE_MODEL="Qwen/Qwen3-14B"
JUDGE_PORT=8731
JUDGE_URL="http://localhost:${JUDGE_PORT}/v1"
OUT_DIR="/scratch/inf0/user/nzukowsk/MASteer/test_calib"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs "${OUT_DIR}"

echo "[$(date)] Job ${SLURM_JOB_ID} — smoke test"
echo "[$(date)] GPUs: $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'n/a')"

# ── 1. Start vLLM judge server (Qwen3-14B on GPU 1+2) ───────────────────────
echo "[$(date)] Starting vLLM judge server (${JUDGE_MODEL}) on port ${JUDGE_PORT} ..."

CUDA_VISIBLE_DEVICES=1,2 \
    "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model          "${JUDGE_MODEL}"           \
        --dtype          bfloat16                   \
        --tensor-parallel-size 2                   \
        --port           "${JUDGE_PORT}"            \
        --disable-log-requests                     \
        --chat-template-text-format string         \
        --max-model-len  4096                      \
        --gpu-memory-utilization 0.90              \
    &
VLLM_PID=$!

# ── 2. Wait until the judge server is ready ──────────────────────────────────
echo "[$(date)] Waiting for vLLM to be ready ..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:${JUDGE_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM ready after ${i}×5s"
        break
    fi
    sleep 5
    if [ $i -eq 60 ]; then
        echo "[$(date)] ERROR: vLLM did not start within 5 minutes"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
done

# ── 3. Kill judge server on exit ─────────────────────────────────────────────
trap 'echo "[$(date)] Killing vLLM (PID $VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true' EXIT

# ── 4. Run calibration on GPU 0 ─────────────────────────────────────────────
echo "[$(date)] Running calibration ..."

CUDA_VISIBLE_DEVICES=0 \
"$PYTHON" -u "${REPO}/scripts/eval_judge_calibration.py" \
    --model          "Qwen/Qwen3-4B"     \
    --raw-vectors-dir "${RAW_VECTORS_DIR}"  \
    --traits          "evil"             \
    --steer-layer     14                 \
    --steer-alpha     1.5                \
    --n-rollouts      2                  \
    --max-prompts     3                  \
    --probe-all-traits                   \
    --output-dir      "${OUT_DIR}"       \
    --device          cuda:0             \
    --judge-model     "${JUDGE_MODEL}"   \
    --judge-base-url  "${JUDGE_URL}"

echo "[$(date)] Done. Results at ${OUT_DIR}/results.jsonl"
