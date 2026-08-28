#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --array=0-52
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/judge_calib_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/judge_calib_%A_%a.err
#SBATCH -t 8:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  Judge Calibration Evaluation — one trait per array job.
#
#  Tests whether the LLM judge's trait scores correlate with internal
#  activation cosine-similarities under CAA steering.
#
#  Per job:
#    - Loads Qwen3-4B (base, no LoRA)
#    - Applies additive CAA steering at --steer-layer for the job's trait
#    - Generates 10 rollouts for each of 100 calibration prompts (1000 total)
#    - Probes each response at the same layer
#    - Scores each response with Grok-4 as judge
#    - Writes results to results/judge_calibration/<trait>/results.jsonl
#
#  All jobs write to separate per-trait output dirs, so they are safe
#  to run concurrently.  The summary script (run after all jobs finish)
#  merges results.jsonl files and computes per-trait correlations.
#
#  GPU layout:
#    cuda:0 — HF model (generation + probing)
#
#  Launch full sweep:
#    sbatch scripts/training/slurm/run_judge_calibration.slurm
#
#  Test a single trait first:
#    sbatch --array=0 scripts/training/slurm/run_judge_calibration.slurm
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
RAW_VECTORS_DIR="/BS/ForNet/work/MASteering/data/vector_extraction/persona/qwen3-4b/bf16"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export XAI_API_KEY="${XAI_API_KEY:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

# ── Trait list (mirrors the persona sweep; one per array task) ───────────────
TRAITS=(
    "adaptable"
    "adaptable-flexible"
    "agreeableness"
    "angry"
    "apathetic"
    "assertive"
    "authentic-dependable"
    "autonomous"
    "boundary-violating-over-intimate"
    "charismatic"
    "collaborative"
    "conscientious"
    "context-aware"
    "creative-playful"
    "curious"
    "directive-stance"
    "emotional-containment"
    "empathy"
    "epistemic-humility"
    "ethical"
    "evil"
    "excessive-validation"
    "exploratory-stance"
    "extraversion"
    "gentle"
    "goal-oriented"
    "hallucinating"
    "humorous"
    "hyperbolic"
    "impolite"
    "interpretive"
    "loquacious"
    "neuroticism"
    "openness"
    "opportunistic"
    "optimistic"
    "over-identification-enmeshment"
    "over-pathologizing"
    "passionate"
    "patient"
    "peacekeeping"
    "premature-reassurance"
    "protocol-rigid-checklist-driven"
    "repair-accountability"
    "resourceful"
    "respectful-attuned"
    "risk-averse-over-referral"
    "rupture-recognition"
    "sarcastic"
    "somber"
    "suggestible-leading"
    "sycophantic"
    "trustworthiness"
)

TRAIT="${TRAITS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="/scratch/inf0/user/nzukowsk/MASteer/judge_calibration/${TRAIT}"

cd "${REPO}"

echo "[$(date)] Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Trait  : ${TRAIT}"
echo "[$(date)] OutDir : ${OUT_DIR}"
echo "[$(date)] GPU    : $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'n/a')"

"$PYTHON" -u "${REPO}/scripts/eval_judge_calibration.py" \
    --model          "Qwen/Qwen3-4B"                     \
    --raw-vectors-dir "${RAW_VECTORS_DIR}"               \
    --traits          "${TRAIT}"                         \
    --steer-layer     14                                 \
    --steer-alpha     15.0                               \
    --n-rollouts      10                                 \
    --max-new-tokens  200                                \
    --temperature     0.8                                \
    --prompts-file    "${REPO}/data/calibration_prompts.json" \
    --output-dir      "${OUT_DIR}"                       \
    --device          cuda:0                             \
    --judge-model     grok-4                             \
    --judge-base-url  "https://api.x.ai/v1"

echo "[$(date)] Done — ${TRAIT}"
