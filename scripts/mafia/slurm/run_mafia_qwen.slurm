#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%5
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/mafia_qwen_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/mafia_qwen_%A_%a.err
#SBATCH -t 06:00:00
#
# SecretMafia noop baseline — Qwen3-14B, TextArena SecretMafia-v0, 8 players.
# 30 episodes sharded over 10 tasks (3 episodes per shard).
# mafia_ratio=0.25 → 2 mafia, 6 civilians, 3 discussion rounds per day.
#
# Submit:
#   sbatch scripts/mafia/slurm/run_mafia_qwen.slurm

set -euo pipefail

NUM_SHARDS=10
RESUME_HOP_LIMIT=20

HOP=${1:-0}
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME} (hop ${HOP})"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

CONFIG="config/games/mafia/mafia_noop_8p.yaml"
SHARD=${SLURM_ARRAY_TASK_ID}

echo "[$(date)] config=${CONFIG} shard=${SHARD}/${NUM_SHARDS}"
"$PYTHON" scripts/run_episode.py --config "${CONFIG}" \
          --shard "${SHARD}" --num-shards "${NUM_SHARDS}"
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} finished this chunk."

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
  sbatch --dependency=afterany:"${SLURM_JOB_ID}" --array="${SLURM_ARRAY_TASK_ID}" \
         "$0" $((HOP + 1))
elif [ "${REMAINING}" -gt 0 ]; then
  echo "[$(date)] WARNING: hop limit (${RESUME_HOP_LIMIT}) reached with ${REMAINING} remaining."
else
  echo "[$(date)] All episodes complete for shard ${SHARD}/${NUM_SHARDS}."
fi
