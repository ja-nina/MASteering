#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-53%10
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/picking_20b_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/picking_20b_%A_%a.err
#SBATCH -t 9:45:00
#
# Picking / Persona sweep — gpt-oss-20b, 3 conditions x 3 player counts x 6 shards
# = 54 array tasks. Replicates Riedl (2025, arXiv 2510.05174) main experiments.
#
#   Flavour 0: plain   2p    Flavour 3: persona 2p    Flavour  6: tom 2p
#   Flavour 1: plain   3p    Flavour 4: persona 3p    Flavour  7: tom 3p
#   Flavour 2: plain  10p    Flavour 5: persona 10p   Flavour  8: tom 10p
#
# Task T -> flavour = T / NUM_SHARDS, shard = T % NUM_SHARDS.
# Same resumable self-chaining scheme as run_gbs_20b.slurm.
#
# Submit:
#   sbatch scripts/run_picking_20b.slurm

set -euo pipefail

NUM_SHARDS=6
RESUME_HOP_LIMIT=20   # N=10 episodes are slow; allow more hops

HOP=${RESUME_HOP:-0}
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME} (hop ${HOP})"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

CONDITIONS=(plain persona tom)
PLAYERS=(2 3 10)

T=${SLURM_ARRAY_TASK_ID}
FLAVOUR=$((T / NUM_SHARDS))
SHARD=$((T % NUM_SHARDS))
CONDITION=${CONDITIONS[$((FLAVOUR / 3))]}
N=${PLAYERS[$((FLAVOUR % 3))]}
CONFIG="config/picking_sweep/gbs_exact_replication_${CONDITION}_${N}p_20b.yaml"

echo "[$(date)] config=${CONFIG} shard=${SHARD}/${NUM_SHARDS}"
"$PYTHON" scripts/run_episode.py --config "${CONFIG}" \
          --shard "${SHARD}" --num-shards "${NUM_SHARDS}"
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} finished this chunk."

REMAINING=$("$PYTHON" - "${CONFIG}" "${SHARD}" "${NUM_SHARDS}" <<'PYEOF'
import glob
import re
import sys

import yaml

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
  echo "[$(date)] ${REMAINING} episode(s) remaining for task ${T} — requeuing (hop $((HOP + 1)))"
  sbatch --dependency=afterany:"${SLURM_JOB_ID}" --array="${T}" \
         --export=ALL,RESUME_HOP=$((HOP + 1)) "$0"
elif [ "${REMAINING}" -gt 0 ]; then
  echo "[$(date)] WARNING: hop limit (${RESUME_HOP_LIMIT}) reached with ${REMAINING} remaining — not requeuing."
else
  echo "[$(date)] All episodes complete for task ${T} (shard ${SHARD}/${NUM_SHARDS})."
fi
