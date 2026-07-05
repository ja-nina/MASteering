#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-17%3
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/gbs_20b_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/gbs_20b_%A_%a.err
#SBATCH -t 9:45:00
#
# GBS sweep — gpt-oss-20b, 6 modes x 3 player counts, 100 episodes each.
#
#   Task  0: noop          2p    Task  3: non_thinking  2p
#   Task  1: noop          3p    Task  4: non_thinking  3p
#   Task  2: noop          4p    Task  5: non_thinking  4p
#
#   Task  6: thinking      2p    Task  9: noop_thinking 2p
#   Task  7: thinking      3p    Task 10: noop_thinking 3p
#   Task  8: thinking      4p    Task 11: noop_thinking 4p
#
#   Task 12: tom           2p    Task 15: tom_thinking  2p
#   Task 13: tom           3p    Task 16: tom_thinking  3p
#   Task 14: tom           4p    Task 17: tom_thinking  4p
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
# 100 episodes per config won't fit in one <10h slot, and each submitted job
# must individually stay under that limit (the limit is per-job, not per
# array). So this is not one long-running script — it's a *chain* of short
# ones: --array=0-17%3 gives you the 18 (mode, player) jobs that can run
# concurrently (3 at a time); on top of that, each individual job in the
# array, when it runs out of its ~9h45 slot before reaching 100 episodes,
# submits its OWN successor job (same task index, new job ID) so that task's
# work continues as a sequence of chunk-jobs, each under the 10h cap.
#
# scripts/run_episode.py is resumable — on startup it skips any
# episode_N.summary.json that already exists, so no completed episode is
# ever overwritten by the next chunk-job. When a task's chunk ends with
# episodes still remaining, this script requeues *only that task* with
# --dependency=afterany on the job that just ran, so the scheduler starts a
# fresh slot right after (whether this one finished cleanly or was killed for
# time) and picks up exactly where it left off. It stops requeuing once all
# episodes are done, or after RESUME_HOP_LIMIT hops as a safety valve against
# a stuck run silently resubmitting forever.
#
# Submit:
#   sbatch scripts/run_gbs_20b.slurm

set -euo pipefail

RESUME_HOP_LIMIT=16   # ~6.5 days of ~9h45 chunks — generous ceiling, not a target

HOP=${RESUME_HOP:-0}
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME} (hop ${HOP})"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

MODES=(noop non_thinking thinking noop_thinking tom tom_thinking)
PLAYERS=(2 3 4)

T=${SLURM_ARRAY_TASK_ID}
MODE=${MODES[$((T / 3))]}
N=${PLAYERS[$((T % 3))]}
CONFIG="config/reasoning_sweep/gbs_${MODE}_${N}p_20b.yaml"

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
