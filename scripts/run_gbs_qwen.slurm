#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-11%3
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/gbs_qwen_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/gbs_qwen_%A_%a.err
#SBATCH -t 9:45:00
#
# GBS sweep — Qwen3-4B, 4 modes x 3 player counts, 100 episodes each.
# Sibling of run_gbs_20b.slurm's tom sweep, run on the smaller model for the
# same 2x2 (ToM prompt x thinking effort) ablation:
#
#   Task  0: noop          2p    Task  6: tom           2p
#   Task  1: noop          3p    Task  7: tom           3p
#   Task  2: noop          4p    Task  8: tom           4p
#
#   Task  3: noop_thinking 2p    Task  9: tom_thinking  2p
#   Task  4: noop_thinking 3p    Task 10: tom_thinking  3p
#   Task  5: noop_thinking 4p    Task 11: tom_thinking  4p
#
# noop_thinking here is intentionally redundant with the pre-existing
# gbs_thinking_{n}p.yaml configs (same enable_thinking=True, same no-prompt
# steering) — it exists under its own run_id purely so the mode-name grid
# lines up with the gpt-oss-20b sweep for analysis. noop / non_thinking /
# thinking (the original 3-mode reasoning sweep) are untouched by this
# script and remain at their original episode counts.
#
# Same resumable-chaining scheme as run_gbs_20b.slurm: each of the 12 array
# tasks below is a *chain* of chunk-jobs, not one long job — every chunk must
# individually stay under the ~9h45 cap, so when a chunk ends with episodes
# still remaining, this script submits its own successor (same task index)
# with --dependency=afterany. scripts/run_episode.py skips episodes that
# already have a summary.json, so nothing gets overwritten across chunks.
# Continues until done or RESUME_HOP_LIMIT hops.
#
# Submit:
#   sbatch scripts/run_gbs_qwen.slurm

set -euo pipefail

RESUME_HOP_LIMIT=16   # ~6.5 days of ~9h45 chunks — generous ceiling, not a target

HOP=${RESUME_HOP:-0}
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME} (hop ${HOP})"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

MODES=(noop noop_thinking tom tom_thinking)
PLAYERS=(2 3 4)

T=${SLURM_ARRAY_TASK_ID}
MODE=${MODES[$((T / 3))]}
N=${PLAYERS[$((T % 3))]}
CONFIG="config/reasoning_sweep/gbs_${MODE}_${N}p.yaml"

echo "[$(date)] config=${CONFIG}"
"$PYTHON" scripts/run_episode.py --config "${CONFIG}"
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} finished this chunk."

REMAINING=$("$PYTHON" - "${CONFIG}" <<'PYEOF'
import glob
import sys

import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
run_dir = f"{cfg['logging']['dir'].rstrip('/')}/{cfg['run_id']}"
done = len(glob.glob(f"{run_dir}/episode_*.summary.json"))
print(max(cfg["episodes"] - done, 0))
PYEOF
)

if [ "${REMAINING}" -gt 0 ] && [ "${HOP}" -lt "${RESUME_HOP_LIMIT}" ]; then
  echo "[$(date)] ${REMAINING} episode(s) remaining for task ${T} — requeuing (hop $((HOP + 1)))"
  sbatch --dependency=afterany:"${SLURM_JOB_ID}" --array="${T}" \
         --export=ALL,RESUME_HOP=$((HOP + 1)) "$0"
elif [ "${REMAINING}" -gt 0 ]; then
  echo "[$(date)] WARNING: hop limit (${RESUME_HOP_LIMIT}) reached with ${REMAINING} episode(s) still remaining for task ${T} — not requeuing. Investigate before resubmitting by hand."
else
  echo "[$(date)] All episodes complete for task ${T}."
fi
