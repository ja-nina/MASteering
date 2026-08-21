#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 10
#SBATCH --gres=gpu:2
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --array=0-43
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/persona_sweep_vllm_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/persona_sweep_vllm_%A_%a.err
#SBATCH -t 12:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  Persona sweep — vLLM accelerated (2 GPUs per job).
#
#  GPU layout per job:
#    cuda:0 — vLLM generation engine (K candidates in ONE batched call)
#    cuda:1 — HF PeftModel (probe_text, recompute_logprobs, grpo_step)
#
#  Speedup vs 1-GPU baseline:
#    ~5-10× faster episode collection (PagedAttention + batched n=K sampling)
#    Weight sync overhead: ~150ms/step (saved to /dev/shm between steps)
#
#  Prerequisite: vLLM must be installed in the conda env.
#    pip install vllm
#
#  Launch:
#    sbatch scripts/training/slurm/run_persona_sweep_vllm.slurm
#
#  To test a single job first:
#    sbatch --array=0 scripts/training/slurm/run_persona_sweep_vllm.slurm
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
CONFIG="${REPO}/configs/training/persona_lora.yaml"

export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME="/BS/conformal-circuits/work/mamba/pkgs/cuda-nvcc-tools-12.9.86-he02047a_2/bin/nvcc"
export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export SVD_BASIS="${REPO}/data/svd_basis/qwen3-4b-residual.pt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vLLM uses its own CUDA process; prevent HF from grabbing both GPUs at import
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

# ── Persona definitions ──────────────────────────────────────────────────────
# Format: "run_name|trait:weight"  — one trait per job
PERSONAS=(
    # maximize (IDs 0-11)
    "max_agreeableness|agreeableness:1.0"
    "max_assertive|assertive:1.0"
    "max_curious|curious:1.0"
    "max_empathy|empathy:1.0"
    "max_extraversion|extraversion:1.0"
    "max_neuroticism|neuroticism:1.0"
    "max_openness|openness:1.0"
    "max_charismatic|charismatic:1.0"
    "max_sarcastic|sarcastic:1.0"
    "max_conscientious|conscientious:1.0"
    "max_angry|angry:1.0"
    "max_ethical|ethical:1.0"
    # minimize (IDs 12-23) — negative weight flips the target direction
    "min_agreeableness|agreeableness:-1.0"
    "min_assertive|assertive:-1.0"
    "min_curious|curious:-1.0"
    "min_empathy|empathy:-1.0"
    "min_extraversion|extraversion:-1.0"
    "min_neuroticism|neuroticism:-1.0"
    "min_openness|openness:-1.0"
    "min_charismatic|charismatic:-1.0"
    "min_sarcastic|sarcastic:-1.0"
    "min_conscientious|conscientious:-1.0"
    "min_angry|angry:-1.0"
    "min_ethical|ethical:-1.0"
    # amoral-roleplay maximize (IDs 24-33)
    "max_evil|evil:1.0"
    "max_directive_stance|directive-stance:1.0"
    "max_epistemic_humility|epistemic-humility:1.0"
    "max_exploratory_stance|exploratory-stance:1.0"
    "max_humorous|humorous:1.0"
    "max_optimistic|optimistic:1.0"
    "max_over_pathologizing|over-pathologizing:1.0"
    "max_protocol_rigid|protocol-rigid-checklist-driven:1.0"
    "max_repair_accountability|repair-accountability:1.0"
    "max_suggestible|suggestible-leading:1.0"
    # amoral-roleplay minimize (IDs 34-43)
    "min_evil|evil:-1.0"
    "min_directive_stance|directive-stance:-1.0"
    "min_epistemic_humility|epistemic-humility:-1.0"
    "min_exploratory_stance|exploratory-stance:-1.0"
    "min_humorous|humorous:-1.0"
    "min_optimistic|optimistic:-1.0"
    "min_over_pathologizing|over-pathologizing:-1.0"
    "min_protocol_rigid|protocol-rigid-checklist-driven:-1.0"
    "min_repair_accountability|repair-accountability:-1.0"
    "min_suggestible|suggestible-leading:-1.0"
)

ENTRY="${PERSONAS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="${ENTRY%%|*}"
TRAITS="${ENTRY##*|}"

cd "${REPO}"

echo "[$(date)] Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] Persona : ${RUN_NAME}"
echo "[$(date)] Traits  : ${TRAITS}"
echo "[$(date)] GPUs    : $(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo 'n/a')"

"$PYTHON" -u "${REPO}/scripts/train_lora.py" \
    --config          "${CONFIG}"                          \
    --traits          "${TRAITS}"                          \
    --run-name        "${RUN_NAME}"                        \
    --local-save-dir  "/scratch/inf0/user/nzukowsk/MASteer/loras" \
    --vllm                                                 \
    --vllm-device     cuda:0                               \
    --train-device    cuda:1                               \
    --vllm-gpu-mem    0.85

echo "[$(date)] Done — ${RUN_NAME}"
