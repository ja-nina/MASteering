#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --array=0-105
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/ipd_nudge_bind_ht_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/ipd_nudge_bind_ht_%A_%a.err
#SBATCH -t 12:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  IPD nudge sweep — bind + high-temperature rollouts, 0 KL penalty.
#
#  Key differences from run_ipd_sweep_vllm_nudge_bind:
#    --temperature 0.9  (was 0.7)  ─┐
#    --top-p       0.95 (was 0.8)   │ more diverse rollouts → higher reward
#    --top-k       50   (was 20)   ─┘ variance → stronger GRPO signal
#
#  GPU layout per job:
#    cuda:0 — vLLM generation engine (K=8 candidates in ONE batched call)
#    cuda:1 — HF PeftModel (probe_text, recompute_logprobs, grpo_step)
#
#  Launch:
#    sbatch scripts/training/slurm/run_ipd_sweep_vllm_nudge_bind_hightemp.slurm
#
#  Test a single trait first:
#    sbatch --array=0 scripts/training/slurm/run_ipd_sweep_vllm_nudge_bind_hightemp.slurm
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
CONFIG="${REPO}/configs/training/ipd_lora.yaml"

export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="/BS/conformal-circuits/work/mamba/pkgs/cuda-nvcc-tools-12.9.86-he02047a_2/bin/nvcc"
export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export SVD_BASIS="${REPO}/data/svd_basis/qwen3-4b-residual.pt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

# ── Persona definitions ──────────────────────────────────────────────────────
# Format: "run_name|trait:weight"
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
TRAITS="${ENTRY##*|}"

cd "${REPO}"

echo "[$(date)] Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Persona : ${RUN_NAME}"
echo "[$(date)] Traits  : ${TRAITS}"
echo "[$(date)] Mode    : nudge + bind + high-temp (T=0.9 top_p=0.95 top_k=50) 0KL"
echo "[$(date)] GPUs    : $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'n/a')"

"$PYTHON" -u "${REPO}/scripts/train_lora.py" \
    --config          "${CONFIG}"                                                  \
    --traits          "${TRAITS}"                                                  \
    --run-name        "ipdNV3_${RUN_NAME}"                                         \
    --local-save-dir  "/scratch/inf0/user/nzukowsk/MASteer/loras_ipd_nudge_v3"    \
    --wandb-project   ma-steering-lora-nothink-v3                                  \
    --vllm                                                                         \
    --vllm-device     cuda:0                                                       \
    --train-device    cuda:1                                                       \
    --nudge                                                                        \
    --bind                                                                         \
    --vllm-gpu-mem    0.85                                                         \
    --temperature     0.9                                                          \
    --top-p           0.95                                                         \
    --top-k           50

echo "[$(date)] Done — ${RUN_NAME}"
