#!/bin/bash
#SBATCH -p gpu20
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --array=0-105%20
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/redteam_baseline_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/redteam_baseline_%A_%a.err
#SBATCH -t 2:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  Red-team baseline sweep — Grok-4 elicits trait from untrained Qwen3-4B.
#
#  Layout per job:
#    cuda:0 — HF base model (no LoRA, no adapters) — inference only
#    xAI API — Grok-4 as adversarial opponent (outbound HTTPS to api.x.ai)
#
#  Each job covers one (trait, direction) pair.
#  Array 0-105: 53 traits × 2 directions (max + min) = 106 jobs.
#  Matches the training sweep PERSONAS list so results align directly.
#
#  XAI_API_KEY is read from ${REPO}/.env via python-dotenv.
#  Make sure .env exists and contains: XAI_API_KEY=<your-key>
#
#  Launch:
#    sbatch scripts/training/slurm/run_redteam_baseline.slurm
#
#  To test a single trait first:
#    sbatch --array=0 scripts/training/slurm/run_redteam_baseline.slurm
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export RAW_VECTORS_DIR="${REPO}/data/vector_extraction/persona/qwen3-4b/bf16"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/redteam_results

# ── Trait × direction definitions ────────────────────────────────────────────
# Format: "trait|direction"  (must match training PERSONAS order)
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
OUT_FILE="/scratch/inf0/user/nzukowsk/MASteer/redteam_results/rt_${TRAIT}_${DIRECTION}.jsonl"

cd "${REPO}"

echo "[$(date)] Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Trait     : ${TRAIT}"
echo "[$(date)] Direction : ${DIRECTION}"
echo "[$(date)] Output    : ${OUT_FILE}"
echo "[$(date)] GPU       : $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'n/a')"

"$PYTHON" -u "${REPO}/scripts/redteam_eval.py" \
    --model              Qwen/Qwen3-4B               \
    --target-trait       "${TRAIT}"                  \
    --direction          "${DIRECTION}"              \
    --raw-vectors-dir    "${RAW_VECTORS_DIR}"        \
    --layer-start        10                          \
    --n-episodes         10                          \
    --game               IteratedPrisonersDilemma-v0 \
    --max-turns          50                          \
    --max-new-tokens     300                         \
    --device             cuda                        \
    --red-team-model     grok-4                      \
    --wandb-project      redteam-baseline            \
    --wandb-run-name     "rt_${TRAIT}_${DIRECTION}"  \
    --output             "${OUT_FILE}"

echo "[$(date)] Done — ${TRAIT} (${DIRECTION})"
