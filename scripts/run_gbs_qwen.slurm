#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-71%3
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/gbs_qwen_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/gbs_qwen_%A_%a.err
#SBATCH -t 9:45:00
#
# GBS sweep — Qwen3-4B, 4 modes x 3 player counts, 100 episodes each, split
# into NUM_SHARDS=6 concurrent shards per (mode, player) flavour — i.e. 12
# flavours x 6 shards = 72 array tasks. Sibling of run_gbs_20b.slurm's tom
# sweep, run on the smaller model for the same 2x2 (ToM prompt x thinking
# effort) ablation. See run_gbs_20b.slurm for why sharding is needed: a
# single task per flavour landed too few episodes per <10h chunk (especially
# at higher player counts) to reach 100 episodes within the chunk-chain hop
# limit.
#
#   Flavour 0: noop          2p    Flavour 6: tom           2p
#   Flavour 1: noop          3p    Flavour 7: tom           3p
#   Flavour 2: noop          4p    Flavour 8: tom           4p
#
#   Flavour 3: noop_thinking 2p    Flavour  9: tom_thinking  2p
#   Flavour 4: noop_thinking 3p    Flavour 10: tom_thinking  3p
#   Flavour 5: noop_thinking 4p    Flavour 11: tom_thinking  4p
#
# Task T -> flavour = T / NUM_SHARDS, shard = T %% NUM_SHARDS.
#
# noop_thinking here is intentionally redundant with the pre-existing
# gbs_thinking_{n}p.yaml configs (same enable_thinking=True, same no-prompt
# steering) — it exists under its own run_id purely so the mode-name grid
# lines up with the gpt-oss-20b sweep for analysis. noop / non_thinking /
# thinking (the original 3-mode reasoning sweep) are untouched by this
# script and remain at their original episode counts.
#
# Same resumable-chaining + sharding scheme as run_gbs_20b.slurm: each shard
# is a chain of chunk-jobs, not one long job — every chunk must individually
# stay under the ~9h45 cap, so when a chunk ends with its assigned episodes
# still incomplete, this script submits its own successor (same task index)
# with --dependency=afterany. scripts/run_episode.py --shard/--num-shards
# skips episodes assigned to other shards and any episode that already has a
# summary.json, so nothing gets overwritten or double-run across chunks or
# shards. Continues until that shard is done or RESUME_HOP_LIMIT hops.
#
# Submit:
#   sbatch scripts/run_gbs_qwen.slurm

set -euo pipefail

NUM_SHARDS=6
RESUME_HOP_LIMIT=16   # ~6.5 days of ~9h45 chunks per shard — generous ceiling, not a target

HOP=${RESUME_HOP:-0}
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME} (hop ${HOP})"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

MODES=(noop noop_thinking tom tom_thinking)
PLAYERS=(2 3 4)

T=${SLURM_ARRAY_TASK_ID}
FLAVOUR=$((T / NUM_SHARDS))
SHARD=$((T % NUM_SHARDS))
MODE=${MODES[$((FLAVOUR / 3))]}
N=${PLAYERS[$((FLAVOUR % 3))]}
CONFIG="config/reasoning_sweep/gbs_${MODE}_${N}p.yaml"

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
  echo "[$(date)] ${REMAINING} episode(s) remaining for task ${T} (shard ${SHARD}/${NUM_SHARDS}) — requeuing (hop $((HOP + 1)))"
  sbatch --dependency=afterany:"${SLURM_JOB_ID}" --array="${T}" \
         --export=ALL,RESUME_HOP=$((HOP + 1)) "$0"
elif [ "${REMAINING}" -gt 0 ]; then
  echo "[$(date)] WARNING: hop limit (${RESUME_HOP_LIMIT}) reached with ${REMAINING} episode(s) still remaining for task ${T} — not requeuing. Investigate before resubmitting by hand."
else
  echo "[$(date)] All episodes complete for task ${T} (shard ${SHARD}/${NUM_SHARDS})."
fi
