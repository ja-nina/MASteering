#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-29%12
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/werewolf_qwen_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/werewolf_qwen_%A_%a.err
#SBATCH -t 06:00:00
#
# Werewolf noop baseline — Qwen3-14B, player-count sweep 6-10.
#
# Layout: PLAYERS x NUM_SHARDS = 5 x 6 = 30 array tasks (0-29).
#   task T → player count PLAYERS[T / NUM_SHARDS], shard T % NUM_SHARDS
#
# To change player counts: edit PLAYERS below AND update --array upper bound:
#   new_upper = len(PLAYERS) * NUM_SHARDS - 1
#
# Submit:
#   sbatch scripts/werewolf/slurm/run_werewolf_qwen.slurm

set -euo pipefail

# ── Configurable sweep ───────────────────────────────────────────────────────
PLAYERS=(6 7 8 9 10)   # player counts to sweep
NUM_SHARDS=6            # array tasks per player count

# ── Requeue guard ────────────────────────────────────────────────────────────
RESUME_HOP_LIMIT=20
HOP=${1:-0}
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME} (hop ${HOP})"

# ── Map array task → player count + shard ────────────────────────────────────
PLAYER_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
SHARD=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))
N=${PLAYERS[$PLAYER_IDX]}
CONFIG="config/games/werewolf/werewolf_noop_${N}p.yaml"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

echo "[$(date)] config=${CONFIG}  players=${N}  shard=${SHARD}/${NUM_SHARDS}"
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
  echo "[$(date)] All episodes complete: ${N}p shard ${SHARD}/${NUM_SHARDS}."
fi
