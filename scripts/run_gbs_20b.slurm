#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-107%3
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/gbs_20b_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/gbs_20b_%A_%a.err
#SBATCH -t 9:45:00
#
# GBS sweep — gpt-oss-20b, 6 modes x 3 player counts, 100 episodes each,
# split into NUM_SHARDS=6 concurrent shards per (mode, player) flavour —
# i.e. 18 flavours x 6 shards = 108 array tasks. One task per flavour was
# too few: higher player counts run slower (more turns/round -> more LLM
# calls per episode), so a single chunk-job only lands 3-4 episodes before
# hitting the ~9h45 cap. At that rate, one task alone would need 25+ chained
# chunk-jobs to reach 100 episodes — past the RESUME_HOP_LIMIT safety cap
# below. Splitting each flavour's 100 episodes across 6 shards (~17 each)
# means each shard's own chunk-chain is short (a handful of hops at most),
# and the shards for one flavour can make progress concurrently instead of
# strictly one-after-another.
#
#   Flavour  0: noop          2p    Flavour  3: non_thinking  2p
#   Flavour  1: noop          3p    Flavour  4: non_thinking  3p
#   Flavour  2: noop          4p    Flavour  5: non_thinking  4p
#
#   Flavour  6: thinking      2p    Flavour  9: noop_thinking 2p
#   Flavour  7: thinking      3p    Flavour 10: noop_thinking 3p
#   Flavour  8: thinking      4p    Flavour 11: noop_thinking 4p
#
#   Flavour 12: tom           2p    Flavour 15: tom_thinking  2p
#   Flavour 13: tom           3p    Flavour 16: tom_thinking  3p
#   Flavour 14: tom           4p    Flavour 17: tom_thinking  4p
#
# Task T -> flavour = T / NUM_SHARDS, shard = T %% NUM_SHARDS. E.g. tasks
# 0-5 are the 6 shards of flavour 0 (noop 2p); tasks 6-11 are flavour 1
# (noop 3p); etc.
#
# noop_thinking: same reasoning effort as `thinking` (system_suffix
# "Reasoning: high") but WITHOUT the step-by-step prompt prefix — isolates
# whether letting the model think helps, separate from telling it to.
#
# tom / tom_thinking: same idea, crossed with a Theory-of-Mind prompt instead
# of the generic step-by-step one (reused verbatim from
# config/gbs_exp_prompt_all_tom.yaml). tom = ToM prompt, no effort override
# (mirrors noop). tom_thinking = ToM prompt + "Reasoning: high" (mirrors
# noop_thinking). Together {noop, noop_thinking, tom, tom_thinking} form a
# 2x2 ToM-prompt x thinking-effort ablation, run identically on Qwen3-4B via
# the equivalent gbs_{mode}_{n}p.yaml configs — see run_gbs_qwen.slurm.
#
# Each shard is itself a *chain* of chunk-jobs, not one long job: a shard's
# ~17 episodes won't always finish in one <10h slot, and every submitted job
# must individually stay under that limit (the limit is per-job, not per
# array or per shard). scripts/run_episode.py --shard/--num-shards assigns
# this shard its disjoint slice of episode indices (ep %% num_shards ==
# shard) and is resumable within that slice — on startup it skips any
# episode_N.summary.json that already exists, so no completed episode is
# ever overwritten by the next chunk-job or by another shard. When a shard's
# chunk ends with its assigned episodes still incomplete, this script
# requeues *only that task* with --dependency=afterany on the job that just
# ran, so the scheduler starts a fresh slot right after (whether this one
# finished cleanly or was killed for time) and picks up exactly where it
# left off. It stops requeuing once that shard's episodes are all done, or
# after RESUME_HOP_LIMIT hops as a safety valve against a stuck run silently
# resubmitting forever.
#
# %3 throttles at most 3 array tasks running at once (adjust to your actual
# concurrent GPU quota — more concurrency here means faster wall-clock
# completion per flavour, since its 6 shards can then run in parallel).
#
# Submit:
#   sbatch scripts/run_gbs_20b.slurm

set -euo pipefail

NUM_SHARDS=6
RESUME_HOP_LIMIT=16   # ~6.5 days of ~9h45 chunks per shard — generous ceiling, not a target

HOP=${RESUME_HOP:-0}
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME} (hop ${HOP})"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

MODES=(noop non_thinking thinking noop_thinking tom tom_thinking)
PLAYERS=(2 3 4)

T=${SLURM_ARRAY_TASK_ID}
FLAVOUR=$((T / NUM_SHARDS))
SHARD=$((T % NUM_SHARDS))
MODE=${MODES[$((FLAVOUR / 3))]}
N=${PLAYERS[$((FLAVOUR % 3))]}
CONFIG="config/reasoning_sweep/gbs_${MODE}_${N}p_20b.yaml"

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
