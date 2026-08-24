#!/bin/bash
#SBATCH -p gpu24
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/lora_ablation_%j.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/lora_ablation_%j.err
#SBATCH -t 6:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  LoRA ablation — persona vs task adapter, per-turn SVD probe metrics
#
#  For each checkpoint found under SAVE_DIR, runs a full game episode under
#  four adapter conditions (base / adapter_a only / adapter_b only / both)
#  and logs per-turn trait activations to JSONL + wandb.
#
#  Usage:
#    # Sweep all checkpoints in a save_dir:
#    sbatch scripts/training/slurm/run_lora_ablation.slurm \
#        outputs/lora/nothink/persona_ipd/max_empathy
#
#    # Single checkpoint:
#    sbatch scripts/training/slurm/run_lora_ablation.slurm \
#        outputs/lora/nothink/persona_ipd/max_empathy/checkpoint_step00500
#
#    # Different game:
#    sbatch scripts/training/slurm/run_lora_ablation.slurm \
#        outputs/lora/nothink/persona_truth_deception/max_empathy \
#        TruthAndDeception-v0
#
#    # Only test task LoRA (skip adapter_a):
#    sbatch scripts/training/slurm/run_lora_ablation.slurm \
#        outputs/lora/nothink/persona_ipd/max_empathy "" --adapter-b-only
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

TARGET="${1:-outputs/lora/nothink/persona_ipd}"   # save_dir OR single ckpt
GAME_ID="${2:-IteratedPrisonersDilemma-v0}"
EXTRA_ARGS="${@:3}"    # e.g. --adapter-b-only, --adapter-a-only

# ── Environment ──────────────────────────────────────────────────────────────
REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export SVD_BASIS="${REPO}/data/svd_basis/qwen3-4b-residual.pt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

ABLATION_DIR="/scratch/inf0/user/nzukowsk/MASteer/ablations"
mkdir -p "/scratch/inf0/user/nzukowsk/MASteer/logs" "${ABLATION_DIR}"

cd "${REPO}"

echo "[$(date)] Job ${SLURM_JOB_ID} — LoRA ablation"
echo "[$(date)] target  : ${TARGET}"
echo "[$(date)] game    : ${GAME_ID}"
echo "[$(date)] extra   : ${EXTRA_ARGS:-<none>}"
echo "[$(date)] GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"

# ── Resolve checkpoints to evaluate ──────────────────────────────────────────
# If TARGET itself looks like a checkpoint (contains adapter_a/ or adapter_b/),
# treat it as a single checkpoint.  Otherwise sweep all checkpoint_step* dirs
# plus final/ under TARGET.

ABS_TARGET="${REPO}/${TARGET}"

is_single_ckpt() {
    # flat:   <ckpt>/adapter_a/adapter_config.json
    # nested: <ckpt>/adapter_a/adapter_a/adapter_config.json
    [ -f "${ABS_TARGET}/adapter_a/adapter_config.json" ]        || \
    [ -f "${ABS_TARGET}/adapter_a/adapter_a/adapter_config.json" ] || \
    [ -f "${ABS_TARGET}/adapter_b/adapter_config.json" ]        || \
    [ -f "${ABS_TARGET}/adapter_b/adapter_b/adapter_config.json" ]
}

if is_single_ckpt; then
    CHECKPOINTS=("${ABS_TARGET}")
else
    mapfile -t STEP_CKPTS < <(ls -d "${ABS_TARGET}"/checkpoint_step* 2>/dev/null | sort)
    FINAL_CKPT=()
    [ -d "${ABS_TARGET}/final" ] && FINAL_CKPT=("${ABS_TARGET}/final")
    CHECKPOINTS=("${STEP_CKPTS[@]}" "${FINAL_CKPT[@]}")
fi

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "[$(date)] ERROR: no checkpoints found under ${ABS_TARGET}"
    exit 1
fi

echo "[$(date)] checkpoints to evaluate: ${#CHECKPOINTS[@]}"
for C in "${CHECKPOINTS[@]}"; do echo "    ${C}"; done

# ── JSONL output — one file per TARGET, accumulates all checkpoints ───────────
RUN_NAME="$(basename "${TARGET}")"
OUTPUT_JSONL="${ABLATION_DIR}/${RUN_NAME}_ablation.jsonl"
echo "[$(date)] output  : ${OUTPUT_JSONL}"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
for CKPT in "${CHECKPOINTS[@]}"; do
    CKPT_NAME="$(basename "${CKPT}")"
    echo "══════════════════════════════════════════════════════"
    echo "[$(date)] ${CKPT_NAME}"
    echo "══════════════════════════════════════════════════════"

    "$PYTHON" -u "${REPO}/scripts/ablate_loras.py"  \
        --checkpoint    "${CKPT}"                   \
        --eval-game     "${GAME_ID}"                \
        --probe-basis   "${SVD_BASIS}"              \
        --probe-layer   35                          \
        --max-turns     20                          \
        --max-new-tokens 512                        \
        --output        "${OUTPUT_JSONL}"           \
        --wandb                                     \
        --wandb-project ma-steering-lora-ablations  \
        --wandb-name    "${RUN_NAME}_${CKPT_NAME}"  \
        ${EXTRA_ARGS}

    echo "[$(date)] done — ${CKPT_NAME}"
    echo ""
done

echo "[$(date)] All checkpoints complete."
echo "[$(date)] Results → ${OUTPUT_JSONL}"
