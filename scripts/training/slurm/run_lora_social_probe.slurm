#!/bin/bash
#SBATCH -p gpu20
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --array=0-105%6
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/social_probe_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/social_probe_%A_%a.err
#SBATCH -t 3:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  LoRA social probe sweep — what does the trainee activate to steer the opp?
#
#  Player 0: no-persona LoRA-adapted Qwen3-4B (trainee)
#  Player 1: same base model, adapter disabled (self-play, no API cost)
#
#  Both players probed on ALL 53 trait directions per turn.
#  Outputs: per-episode trainee/opponent cos sims + social grammar table.
#
#  Array 0-105: 53 traits × 2 directions = 106 jobs (same layout as training).
#  Each job loads the latest checkpoint from the no-persona LoRA directory.
#
#  Launch:
#    sbatch scripts/training/slurm/run_lora_social_probe.slurm
#
#  Single test:
#    sbatch --array=0 scripts/training/slurm/run_lora_social_probe.slurm
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LORA_ROOT="/scratch/inf0/user/nzukowsk/MASteer/loras_ipd_nudge_v3_nopersona"
PVEC_ROOT="/scratch/inf0/user/nzukowsk/PEVectors/PEVectors/data/vector_extraction/persona"
STD_VECTORS_DIR="${PVEC_ROOT}/qwen3-4b/bf16"
AMORAL_VECTORS_DIR="${PVEC_ROOT}/qwen3-4b-amoral-roleplay/bf16"

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/social_probe_results
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/social_probe_activations

# ── Trait × direction definitions (same order as training sweep) ─────────────
ENTRIES=(
    "ethical|max"
    "ethical|min"
    "evil|max"
    "evil|min"
    "adaptable|max"
    "adaptable|min"
    "adaptable-flexible|max"
    "adaptable-flexible|min"
    "agreeableness|max"
    "agreeableness|min"
    "angry|max"
    "angry|min"
    "apathetic|max"
    "apathetic|min"
    "assertive|max"
    "assertive|min"
    "authentic-dependable|max"
    "authentic-dependable|min"
    "autonomous|max"
    "autonomous|min"
    "boundary-violating-over-intimate|max"
    "boundary-violating-over-intimate|min"
    "charismatic|max"
    "charismatic|min"
    "collaborative|max"
    "collaborative|min"
    "conscientious|max"
    "conscientious|min"
    "context-aware|max"
    "context-aware|min"
    "creative-playful|max"
    "creative-playful|min"
    "curious|max"
    "curious|min"
    "directive-stance|max"
    "directive-stance|min"
    "emotional-containment|max"
    "emotional-containment|min"
    "empathy|max"
    "empathy|min"
    "epistemic-humility|max"
    "epistemic-humility|min"
    "excessive-validation|max"
    "excessive-validation|min"
    "exploratory-stance|max"
    "exploratory-stance|min"
    "extraversion|max"
    "extraversion|min"
    "gentle|max"
    "gentle|min"
    "goal-oriented|max"
    "goal-oriented|min"
    "hallucinating|max"
    "hallucinating|min"
    "humorous|max"
    "humorous|min"
    "hyperbolic|max"
    "hyperbolic|min"
    "impolite|max"
    "impolite|min"
    "interpretive|max"
    "interpretive|min"
    "loquacious|max"
    "loquacious|min"
    "neuroticism|max"
    "neuroticism|min"
    "openness|max"
    "openness|min"
    "opportunistic|max"
    "opportunistic|min"
    "optimistic|max"
    "optimistic|min"
    "over-identification-enmeshment|max"
    "over-identification-enmeshment|min"
    "over-pathologizing|max"
    "over-pathologizing|min"
    "passionate|max"
    "passionate|min"
    "patient|max"
    "patient|min"
    "peacekeeping|max"
    "peacekeeping|min"
    "premature-reassurance|max"
    "premature-reassurance|min"
    "protocol-rigid-checklist-driven|max"
    "protocol-rigid-checklist-driven|min"
    "repair-accountability|max"
    "repair-accountability|min"
    "resourceful|max"
    "resourceful|min"
    "respectful-attuned|max"
    "respectful-attuned|min"
    "risk-averse-over-referral|max"
    "risk-averse-over-referral|min"
    "rupture-recognition|max"
    "rupture-recognition|min"
    "sarcastic|max"
    "sarcastic|min"
    "somber|max"
    "somber|min"
    "suggestible-leading|max"
    "suggestible-leading|min"
    "sycophantic|max"
    "sycophantic|min"
    "trustworthiness|max"
    "trustworthiness|min"
)

ENTRY="${ENTRIES[$SLURM_ARRAY_TASK_ID]}"
TRAIT="${ENTRY%%|*}"
DIRECTION="${ENTRY##*|}"

# LoRA dir uses underscores for trait names
TRAIT_SLUG="${TRAIT//-/_}"
LORA_DIR="${LORA_ROOT}/ipdNV3np_${DIRECTION}_${TRAIT_SLUG}"
OUT_FILE="/scratch/inf0/user/nzukowsk/MASteer/social_probe_results/sp_${TRAIT}_${DIRECTION}.jsonl"

if [ ! -d "${LORA_DIR}" ]; then
    echo "[$(date)] ERROR: LoRA dir not found: ${LORA_DIR}"
    exit 1
fi

cd "${REPO}"

echo "[$(date)] Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Trait     : ${TRAIT}"
echo "[$(date)] Direction : ${DIRECTION}"
echo "[$(date)] LoRA dir  : ${LORA_DIR}"
echo "[$(date)] Output    : ${OUT_FILE}"
echo "[$(date)] GPU       : $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'n/a')"

"$PYTHON" -u "${REPO}/scripts/lora_social_probe.py" \
    --model              Qwen/Qwen3-4B                \
    --lora-dir           "${LORA_DIR}"                \
    --target-trait       "${TRAIT}"                   \
    --direction          "${DIRECTION}"               \
    --std-vectors-dir    "${STD_VECTORS_DIR}"         \
    --amoral-vectors-dir "${AMORAL_VECTORS_DIR}"      \
    --layer-start        20                           \
    --n-episodes         10                           \
    --game               IteratedPrisonersDilemma-v0  \
    --max-turns          50                           \
    --max-new-tokens     300                          \
    --device             cuda                         \
    --wandb-project      lora-social-probe                                      \
    --wandb-run-name     "sp_${TRAIT}_${DIRECTION}"                            \
    --output             "${OUT_FILE}"                                         \
    --activations-dir    /scratch/inf0/user/nzukowsk/MASteer/social_probe_activations \
    --full-token-episodes 1

echo "[$(date)] Done — ${TRAIT} (${DIRECTION})"
