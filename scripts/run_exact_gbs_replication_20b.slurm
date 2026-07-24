#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-53%10
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/picking_20b_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/picking_20b_%A_%a.err
#SBATCH -t 23:00:00
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06
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
#   sbatch scripts/run_exact_gbs_replication_20b.slurm

# -e removed: we need the script to reach the requeue block even if Python
# exits non-zero (killed at walltime, OOM, etc.)
set -uo pipefail

NUM_SHARDS=6
RESUME_HOP_LIMIT=20

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

# ── requeue helper ────────────────────────────────────────────────────────────
_requeue() {
  REMAINING=$("$PYTHON" - "${CONFIG}" "${SHARD}" "${NUM_SHARDS}" <<'PYEOF'
import glob, re, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
shard, num_shards = int(sys.argv[2]), int(sys.argv[3])
run_dir = f"{cfg['logging']['dir'].rstrip('/')}/{cfg['run_id']}"
done = {int(re.search(r"episode_(\d+)\.summary\.json$", p).group(1))
        for p in glob.glob(f"{run_dir}/episode_*.summary.json")
        if re.search(r"episode_(\d+)\.summary\.json$", p)}
assigned = [ep for ep in range(cfg["episodes"]) if ep % num_shards == shard]
print(sum(1 for ep in assigned if ep not in done))
PYEOF
  ) || REMAINING=0

  if [ "${REMAINING}" -gt 0 ] && [ "${HOP}" -lt "${RESUME_HOP_LIMIT}" ]; then
    echo "[$(date)] ${REMAINING} episode(s) remaining — requeuing (hop $((HOP + 1)))"
    sbatch --dependency=afterany:"${SLURM_JOB_ID}" --array="${T}" \
           --export=ALL,RESUME_HOP=$((HOP + 1)) "$0"
  elif [ "${REMAINING}" -gt 0 ]; then
    echo "[$(date)] WARNING: hop limit (${RESUME_HOP_LIMIT}) reached with ${REMAINING} remaining."
  else
    echo "[$(date)] All episodes complete for task ${T} (shard ${SHARD}/${NUM_SHARDS})."
  fi
}

# ── SIGTERM trap (SLURM walltime kill) ───────────────────────────────────────
# Run Python in background so we can intercept SIGTERM before SIGKILL arrives.
# When SLURM signals the job, we forward TERM to Python, wait for it to flush
# the current episode, then run the requeue check and exit cleanly.
_on_term() {
  echo "[$(date)] SIGTERM received (walltime?) — stopping Python and requeuing."
  kill -TERM "${PYTHON_PID}" 2>/dev/null || true
  wait "${PYTHON_PID}" || true
  _requeue
  exit 0
}
trap '_on_term' TERM

echo "[$(date)] config=${CONFIG} shard=${SHARD}/${NUM_SHARDS}"
"$PYTHON" scripts/run_episode.py --config "${CONFIG}" \
          --shard "${SHARD}" --num-shards "${NUM_SHARDS}" &
PYTHON_PID=$!
wait "${PYTHON_PID}"
PYTHON_EXIT=$?

echo "[$(date)] Python finished (exit ${PYTHON_EXIT})."
_requeue
