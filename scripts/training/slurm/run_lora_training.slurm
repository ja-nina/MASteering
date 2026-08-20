#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/lora_train_%j.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/lora_train_%j.err
#SBATCH -t 24:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  Dual persona LoRA training — REINFORCE self-play
#  adapter_a  (maximiser): SVD persona LoRA, frozen A-matrix, trains B only
#  adapter_b  (minimiser): standard LoRA, trains toward negated trait target
# ════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   sbatch scripts/training/slurm/run_lora_training.slurm
#   sbatch scripts/training/slurm/run_lora_training.slurm configs/training/persona_lora.yaml
#   sbatch scripts/training/slurm/run_lora_training.slurm configs/training/persona_lora.yaml --no-persona-lora
#   sbatch scripts/training/slurm/run_lora_training.slurm configs/training/persona_lora.yaml --rank 8
#
# Outputs land in save_dir from the config (default: outputs/lora/...).
# Checkpoints: <save_dir>/checkpoint_ep<N>/adapter_{a,b}/
# Final:       <save_dir>/final/adapter_{a,b}/
# Log:         <save_dir>/training_log.jsonl

set -euo pipefail

CONFIG="${1:-configs/training/persona_lora.yaml}"
EXTRA_ARGS="${@:2}"   # any additional flags (--no-persona-lora, --rank N, ...)

# ── Environment ────────────────────────────────────────────────────────────
REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export SVD_BASIS="${SVD_BASIS:-${REPO}/data/svd_basis/qwen3-4b-residual.pt}"
# Reduce memory fragmentation — important when holding multiple LoRA grad graphs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

cd "${REPO}"

echo "[$(date)] Job ${SLURM_JOB_ID} — LoRA training"
echo "[$(date)] Config: ${CONFIG}"
echo "[$(date)] Extra args: ${EXTRA_ARGS:-<none>}"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"

# ── Run ────────────────────────────────────────────────────────────────────
"$PYTHON" -u "${REPO}/scripts/train_lora.py" \
    --config "${CONFIG}" \
    ${EXTRA_ARGS}

echo "[$(date)] Training complete."
