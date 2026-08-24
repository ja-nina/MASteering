#!/bin/bash
#SBATCH -p gpu24
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --array=0-105
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/lora_ablation_traits_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/lora_ablation_traits_%A_%a.err
#SBATCH -t 6:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  LoRA ablation — all 106 persona traits × nudge prompt × step-50 checkpoint
#
#  Each array task = one persona, running the 4-condition ablation (base /
#  adapter_a / adapter_b / both) with that persona's nudge prompt as the
#  trainee system prompt.  All results accumulate into one JSONL file.
#
#  Usage:
#    # Full sweep
#    sbatch scripts/training/slurm/run_lora_ablation_all_traits.slurm
#
#    # Single trait first (e.g. empathy = index 38)
#    sbatch --array=38 scripts/training/slurm/run_lora_ablation_all_traits.slurm
#
#    # Override checkpoint (default: checkpoint_step00050 inside SAVE_DIR)
#    CKPT=/scratch/.../loras_ipd_nudge/ipdN_max_empathy/checkpoint_step00050 \
#    sbatch scripts/training/slurm/run_lora_ablation_all_traits.slurm
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

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

# ── Persona definitions (mirrors run_ipd_sweep_vllm_nudge.slurm) ─────────────
PERSONAS=(
    "max_ethical|ethical:1.0"
    "min_ethical|ethical:-1.0"
    "max_evil|evil:1.0"
    "min_evil|evil:-1.0"
    "max_adaptable|adaptable:1.0"
    "min_adaptable|adaptable:-1.0"
    "max_adaptable_flexible|adaptable-flexible:1.0"
    "min_adaptable_flexible|adaptable-flexible:-1.0"
    "max_agreeableness|agreeableness:1.0"
    "min_agreeableness|agreeableness:-1.0"
    "max_angry|angry:1.0"
    "min_angry|angry:-1.0"
    "max_apathetic|apathetic:1.0"
    "min_apathetic|apathetic:-1.0"
    "max_assertive|assertive:1.0"
    "min_assertive|assertive:-1.0"
    "max_authentic_dependable|authentic-dependable:1.0"
    "min_authentic_dependable|authentic-dependable:-1.0"
    "max_autonomous|autonomous:1.0"
    "min_autonomous|autonomous:-1.0"
    "max_boundary_violating|boundary-violating-over-intimate:1.0"
    "min_boundary_violating|boundary-violating-over-intimate:-1.0"
    "max_charismatic|charismatic:1.0"
    "min_charismatic|charismatic:-1.0"
    "max_collaborative|collaborative:1.0"
    "min_collaborative|collaborative:-1.0"
    "max_conscientious|conscientious:1.0"
    "min_conscientious|conscientious:-1.0"
    "max_context_aware|context-aware:1.0"
    "min_context_aware|context-aware:-1.0"
    "max_creative_playful|creative-playful:1.0"
    "min_creative_playful|creative-playful:-1.0"
    "max_curious|curious:1.0"
    "min_curious|curious:-1.0"
    "max_directive_stance|directive-stance:1.0"
    "min_directive_stance|directive-stance:-1.0"
    "max_emotional_containment|emotional-containment:1.0"
    "min_emotional_containment|emotional-containment:-1.0"
    "max_empathy|empathy:1.0"
    "min_empathy|empathy:-1.0"
    "max_epistemic_humility|epistemic-humility:1.0"
    "min_epistemic_humility|epistemic-humility:-1.0"
    "max_excessive_validation|excessive-validation:1.0"
    "min_excessive_validation|excessive-validation:-1.0"
    "max_exploratory_stance|exploratory-stance:1.0"
    "min_exploratory_stance|exploratory-stance:-1.0"
    "max_extraversion|extraversion:1.0"
    "min_extraversion|extraversion:-1.0"
    "max_gentle|gentle:1.0"
    "min_gentle|gentle:-1.0"
    "max_goal_oriented|goal-oriented:1.0"
    "min_goal_oriented|goal-oriented:-1.0"
    "max_hallucinating|hallucinating:1.0"
    "min_hallucinating|hallucinating:-1.0"
    "max_humorous|humorous:1.0"
    "min_humorous|humorous:-1.0"
    "max_hyperbolic|hyperbolic:1.0"
    "min_hyperbolic|hyperbolic:-1.0"
    "max_impolite|impolite:1.0"
    "min_impolite|impolite:-1.0"
    "max_interpretive|interpretive:1.0"
    "min_interpretive|interpretive:-1.0"
    "max_loquacious|loquacious:1.0"
    "min_loquacious|loquacious:-1.0"
    "max_neuroticism|neuroticism:1.0"
    "min_neuroticism|neuroticism:-1.0"
    "max_openness|openness:1.0"
    "min_openness|openness:-1.0"
    "max_opportunistic|opportunistic:1.0"
    "min_opportunistic|opportunistic:-1.0"
    "max_optimistic|optimistic:1.0"
    "min_optimistic|optimistic:-1.0"
    "max_over_identification|over-identification-enmeshment:1.0"
    "min_over_identification|over-identification-enmeshment:-1.0"
    "max_over_pathologizing|over-pathologizing:1.0"
    "min_over_pathologizing|over-pathologizing:-1.0"
    "max_passionate|passionate:1.0"
    "min_passionate|passionate:-1.0"
    "max_patient|patient:1.0"
    "min_patient|patient:-1.0"
    "max_peacekeeping|peacekeeping:1.0"
    "min_peacekeeping|peacekeeping:-1.0"
    "max_premature_reassurance|premature-reassurance:1.0"
    "min_premature_reassurance|premature-reassurance:-1.0"
    "max_protocol_rigid|protocol-rigid-checklist-driven:1.0"
    "min_protocol_rigid|protocol-rigid-checklist-driven:-1.0"
    "max_repair_accountability|repair-accountability:1.0"
    "min_repair_accountability|repair-accountability:-1.0"
    "max_resourceful|resourceful:1.0"
    "min_resourceful|resourceful:-1.0"
    "max_respectful_attuned|respectful-attuned:1.0"
    "min_respectful_attuned|respectful-attuned:-1.0"
    "max_risk_averse|risk-averse-over-referral:1.0"
    "min_risk_averse|risk-averse-over-referral:-1.0"
    "max_rupture_recognition|rupture-recognition:1.0"
    "min_rupture_recognition|rupture-recognition:-1.0"
    "max_sarcastic|sarcastic:1.0"
    "min_sarcastic|sarcastic:-1.0"
    "max_somber|somber:1.0"
    "min_somber|somber:-1.0"
    "max_suggestible|suggestible-leading:1.0"
    "min_suggestible|suggestible-leading:-1.0"
    "max_sycophantic|sycophantic:1.0"
    "min_sycophantic|sycophantic:-1.0"
    "max_trustworthiness|trustworthiness:1.0"
    "min_trustworthiness|trustworthiness:-1.0"
)

ENTRY="${PERSONAS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="${ENTRY%%|*}"
TRAIT_SPEC="${ENTRY##*|}"

# Checkpoint: default to step-50 inside the nudge save dir,
# or override via env var CKPT.
CKPT="${CKPT:-/scratch/inf0/user/nzukowsk/MASteer/loras_ipd_nudge/ipdN_${RUN_NAME}/checkpoint_step00050}"

echo "[$(date)] Job ${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] persona  : ${RUN_NAME}"
echo "[$(date)] trait    : ${TRAIT_SPEC}"
echo "[$(date)] checkpoint: ${CKPT}"
echo "[$(date)] GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/a')"

OUTPUT_JSONL="${ABLATION_DIR}/all_traits_step50_ablation.jsonl"
echo "[$(date)] output   : ${OUTPUT_JSONL}"

"$PYTHON" -u "${REPO}/scripts/ablate_loras.py"          \
    --checkpoint    "${CKPT}"                            \
    --eval-game     IteratedPrisonersDilemma-v0          \
    --probe-basis   "${SVD_BASIS}"                       \
    --probe-layer   35                                   \
    --nudge-trait   "${TRAIT_SPEC}"                      \
    --max-turns     20                                   \
    --n-episodes    5                                    \
    --max-new-tokens 512                                 \
    --output        "${OUTPUT_JSONL}"                    \
    --wandb                                              \
    --wandb-project ma-steering-lora-ablations           \
    --wandb-name    "step50_${RUN_NAME}"

echo "[$(date)] Done — ${RUN_NAME}"
