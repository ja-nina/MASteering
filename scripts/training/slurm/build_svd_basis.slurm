#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/build_svd_basis_%j.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/build_svd_basis_%j.err
#SBATCH -t 02:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  Build SVD persona basis  —  standard traits + 12 selected amoral traits
#
#  Standard traits:  $STD_DIR  (all *_raw.pt files)
#  Amoral traits:    $AMORAL_DIR  (only the 12 slugs listed in AMORAL_SLUGS)
#  The 90%-cosine-sim deduplication merges near-identical directions from
#  both sets, so no manual merging step is needed.
#
#  Output: $OUT_PT  — consumed by SVDPersonaProbe and SVDSteering
#
# Usage:
#   sbatch scripts/training/slurm/build_svd_basis.slurm
#
# After the job finishes verify the basis:
#   python -c "
#   import torch
#   b = torch.load('$OUT_PT', weights_only=False)
#   print('slugs:', b['slugs'])
#   print('layers:', sorted(b['Vk'].keys()))
#   print('k (SVD rank):', next(iter(b['Vk'].values())).shape[0])
#   "
# ════════════════════════════════════════════════════════════════════════════

set -euo pipefail

REPO="/BS/ForNet/work/MASteering"
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

# ── Paths ────────────────────────────────────────────────────────────────────
STD_DIR="${REPO}/data/vector_extraction/persona/qwen3-4b/bf16"
AMORAL_DIR="${REPO}/data/vector_extraction/persona/qwen3-4b-amoral-roleplay"
OUT_PT="${REPO}/data/svd_basis/qwen3-4b-attn-with-amoral.pt"

# ── The 12 amoral traits to include ─────────────────────────────────────────
# Replace these placeholders with the actual slug names from $AMORAL_DIR.
# Run `ls $AMORAL_DIR/*_raw.pt | sed 's/.*\///;s/_raw.pt//'` to see all slugs.
AMORAL_SLUGS=(
    "SLUG_1"
    "SLUG_2"
    "SLUG_3"
    "SLUG_4"
    "SLUG_5"
    "SLUG_6"
    "SLUG_7"
    "SLUG_8"
    "SLUG_9"
    "SLUG_10"
    "SLUG_11"
    "SLUG_12"
)

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs
mkdir -p "$(dirname ${OUT_PT})"

echo "[$(date)] Job ${SLURM_JOB_ID} — building SVD basis"
echo "[$(date)] std-dir:    ${STD_DIR}"
echo "[$(date)] amoral-dir: ${AMORAL_DIR}"
echo "[$(date)] amoral slugs: ${AMORAL_SLUGS[*]}"
echo "[$(date)] out: ${OUT_PT}"

cd "${REPO}"

"$PYTHON" scripts/build_svd_basis.py \
    --std-dir    "${STD_DIR}" \
    --amoral-dir "${AMORAL_DIR}" \
    --amoral-slugs "${AMORAL_SLUGS[@]}" \
    --hook       attn \
    --model      qwen3-4b \
    --out        "${OUT_PT}"

echo "[$(date)] Done. Verifying output..."

"$PYTHON" - <<PYEOF
import torch
b = torch.load("${OUT_PT}", weights_only=False)
print(f"  slugs ({len(b['slugs'])}):", b['slugs'])
print(f"  layers: {sorted(b['Vk'].keys())}")
k = next(iter(b['Vk'].values())).shape[0]
print(f"  SVD rank k={k}")
PYEOF

echo "[$(date)] Basis ready at ${OUT_PT}"
