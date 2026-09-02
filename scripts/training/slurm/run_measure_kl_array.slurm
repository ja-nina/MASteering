#!/bin/bash
#SBATCH -p gpu20
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --array=0-105%20
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/measure_kl_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/measure_kl_%A_%a.err
#SBATCH -t 1:30:00
#
# ════════════════════════════════════════════════════════════════════════════
#  KL calibration — array sweep (one job per LoRA checkpoint)
#
#  Array 0-105: 53 traits × 2 directions = 106 jobs.
#  Each job measures KL(LoRA || base) for one checkpoint and saves to its
#  own JSON file under kl_results/.
#
#  Launch:
#    sbatch scripts/training/slurm/run_measure_kl_array.slurm
#
#  Single test:
#    sbatch --array=0 scripts/training/slurm/run_measure_kl_array.slurm
#
#  After all jobs finish, aggregate:
#    python scripts/aggregate_kl.py \
#        /scratch/inf0/user/nzukowsk/MASteer/kl_results/ \
#        --output /scratch/inf0/user/nzukowsk/MASteer/kl_calibration.json
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LORA_ROOT="/scratch/inf0/user/nzukowsk/MASteer/loras_ipd_nudge_v3_nopersona"
OUT_DIR="/scratch/inf0/user/nzukowsk/MASteer/kl_results"

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs
mkdir -p "${OUT_DIR}"

# ── same trait × direction table as the training sweep ───────────────────────
ENTRIES=(
    "ethical|max"           "ethical|min"
    "evil|max"              "evil|min"
    "adaptable|max"         "adaptable|min"
    "adaptable-flexible|max" "adaptable-flexible|min"
    "agreeableness|max"     "agreeableness|min"
    "angry|max"             "angry|min"
    "apathetic|max"         "apathetic|min"
    "assertive|max"         "assertive|min"
    "authentic-dependable|max" "authentic-dependable|min"
    "autonomous|max"        "autonomous|min"
    "boundary-violating-over-intimate|max" "boundary-violating-over-intimate|min"
    "charismatic|max"       "charismatic|min"
    "collaborative|max"     "collaborative|min"
    "conscientious|max"     "conscientious|min"
    "context-aware|max"     "context-aware|min"
    "creative-playful|max"  "creative-playful|min"
    "curious|max"           "curious|min"
    "directive-stance|max"  "directive-stance|min"
    "emotional-containment|max" "emotional-containment|min"
    "empathy|max"           "empathy|min"
    "epistemic-humility|max" "epistemic-humility|min"
    "excessive-validation|max" "excessive-validation|min"
    "exploratory-stance|max" "exploratory-stance|min"
    "extraversion|max"      "extraversion|min"
    "gentle|max"            "gentle|min"
    "goal-oriented|max"     "goal-oriented|min"
    "hallucinating|max"     "hallucinating|min"
    "humorous|max"          "humorous|min"
    "hyperbolic|max"        "hyperbolic|min"
    "impolite|max"          "impolite|min"
    "interpretive|max"      "interpretive|min"
    "loquacious|max"        "loquacious|min"
    "neuroticism|max"       "neuroticism|min"
    "openness|max"          "openness|min"
    "opportunistic|max"     "opportunistic|min"
    "optimistic|max"        "optimistic|min"
    "over-identification-enmeshment|max" "over-identification-enmeshment|min"
    "over-pathologizing|max" "over-pathologizing|min"
    "passionate|max"        "passionate|min"
    "patient|max"           "patient|min"
    "peacekeeping|max"      "peacekeeping|min"
    "premature-reassurance|max" "premature-reassurance|min"
    "protocol-rigid-checklist-driven|max" "protocol-rigid-checklist-driven|min"
    "repair-accountability|max" "repair-accountability|min"
    "resourceful|max"       "resourceful|min"
    "respectful-attuned|max" "respectful-attuned|min"
    "risk-averse-over-referral|max" "risk-averse-over-referral|min"
    "rupture-recognition|max" "rupture-recognition|min"
    "sarcastic|max"         "sarcastic|min"
    "somber|max"            "somber|min"
    "suggestible-leading|max" "suggestible-leading|min"
    "sycophantic|max"       "sycophantic|min"
    "trustworthiness|max"   "trustworthiness|min"
)

ENTRY="${ENTRIES[$SLURM_ARRAY_TASK_ID]}"
TRAIT="${ENTRY%%|*}"
DIRECTION="${ENTRY##*|}"
TRAIT_SLUG="${TRAIT//-/_}"

LORA_DIR="${LORA_ROOT}/ipdNV3np_${DIRECTION}_${TRAIT_SLUG}"
OUT_FILE="${OUT_DIR}/kl_${TRAIT}_${DIRECTION}.json"

if [ ! -d "${LORA_DIR}" ]; then
    echo "[$(date)] ERROR: LoRA dir not found: ${LORA_DIR}"
    exit 1
fi

cd "${REPO}"

echo "[$(date)] Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Trait     : ${TRAIT} (${DIRECTION})"
echo "[$(date)] LoRA dir  : ${LORA_DIR}"
echo "[$(date)] Output    : ${OUT_FILE}"
echo "[$(date)] GPU       : $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'n/a')"

"$PYTHON" -u "${REPO}/scripts/measure_kl.py" \
    --model          Qwen/Qwen3-4B            \
    --lora-dir       "${LORA_DIR}"            \
    --n-samples      20                       \
    --max-new-tokens 1024                     \
    --temperature    0.7                      \
    --device         cuda                     \
    --dtype          bfloat16                 \
    --output         "${OUT_FILE}"

echo "[$(date)] Done — ${TRAIT} (${DIRECTION})"
