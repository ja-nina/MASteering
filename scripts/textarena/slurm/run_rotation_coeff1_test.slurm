#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --array=0-4%5
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/rotation_coeff1_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/rotation_coeff1_%A_%a.err
#SBATCH -t 04:00:00
#
# Sanity-check run: rotation steering at coefficient=1.0 rad (57deg).
# Single config — mafia opportunistic — 5 shards × 50 episodes.
# Logs named rotation_coeff1_* to distinguish from the main rotation sweep (coeff=0.3).
#
# Submit:
#   sbatch scripts/textarena/slurm/run_rotation_coeff1_test.slurm

set -euo pipefail

NUM_SHARDS=5
RESUME_HOP_LIMIT=10
HOP=${1:-0}

SHARD=${SLURM_ARRAY_TASK_ID}
CONFIG="config/games/mafia/mafia_activation_wolf_opportunistic_rotation_coeff1_8p.yaml"

echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} → shard ${SHARD} (hop ${HOP})"
echo "[$(date)] Config: ${CONFIG}"

# ── Environment ────────────────────────────────────────────────────────────
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export PERSONA_VECTORS_ROOT="/scratch/inf0/user/nzukowsk/PEVectors/PEVectors/data/vector_extraction/persona/qwen3-14b"

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

# ── Run ────────────────────────────────────────────────────────────────────
"$PYTHON" scripts/run_episode.py \
    --config    "${CONFIG}" \
    --shard     "${SHARD}" \
    --num-shards "${NUM_SHARDS}"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} finished."

# ── Self-requeue if episodes remain ───────────────────────────────────────
REMAINING=$("$PYTHON" - "${CONFIG}" "${SHARD}" "${NUM_SHARDS}" <<'PYEOF'
import glob, re, sys, yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
shard, num_shards = int(sys.argv[2]), int(sys.argv[3])
run_dir = f"{cfg['logging']['dir'].rstrip('/')}/{cfg['run_id']}"

done = set()
for p in glob.glob(f"{run_dir}/episode_*.summary.json"):
    m = re.search(r"episode_(\d+)\.summary\.json$", p)
    if m:
        done.add(int(m.group(1)))

assigned = [ep for ep in range(cfg["episodes"]) if ep % num_shards == shard]
print(sum(1 for ep in assigned if ep not in done))
PYEOF
)

if [ "${REMAINING}" -gt 0 ] && [ "${HOP}" -lt "${RESUME_HOP_LIMIT}" ]; then
    echo "[$(date)] ${REMAINING} episode(s) remaining — requeuing (hop $((HOP + 1)))"
    sbatch --dependency=afterany:"${SLURM_JOB_ID}" \
           --array="${SLURM_ARRAY_TASK_ID}" \
           "$0" $((HOP + 1))
elif [ "${REMAINING}" -gt 0 ]; then
    echo "[$(date)] WARNING: hop limit reached with ${REMAINING} remaining."
else
    echo "[$(date)] All episodes complete for shard ${SHARD}."
fi
