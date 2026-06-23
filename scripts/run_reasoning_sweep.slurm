#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-29%9
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/reasoning_sweep_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/reasoning_sweep_%A_%a.err
#SBATCH -t 4:00:00
#
# Reasoning mode sweep
#
#   Tasks  0-17 : Qwen/Qwen3-4B — noop | non_thinking | thinking  x  gbs | beauty_contest  x  2 | 3 | 4 players
#   Tasks 18-29 : gpt-oss-20b   —       non_thinking  | thinking  x  gbs | beauty_contest  x  2 | 3 | 4 players
#
# Prerequisites:
#   python scripts/gen_reasoning_sweep_configs.py
#
# Submit all:
#   sbatch scripts/run_reasoning_sweep.slurm
#
# 4B only:
#   sbatch --array=0-17%9 scripts/run_reasoning_sweep.slurm
#
# 20B only:
#   sbatch --array=18-29%6 scripts/run_reasoning_sweep.slurm

set -euo pipefail
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME}"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

T=${SLURM_ARRAY_TASK_ID}

if [ "${T}" -lt 18 ]; then
  # ── Qwen3-4B: 3 modes x 2 games x 3 players (tasks 0-17) ──────────────────
#   MODES=(noop non_thinking thinking)
#   GAMES=(gbs beauty_contest)
#   PLAYERS=(2 3 4)
#   MODE=${MODES[$((T / 6))]}
#   GAME=${GAMES[$(( (T / 3) % 2 ))]}
#   N=${PLAYERS[$((T % 3))]}
#   CONFIG="config/reasoning_sweep/${GAME}_${MODE}_${N}p.yaml"
# else
  # ── gpt-oss-20b: 2 modes x 2 games x 3 players (tasks 18-29) ──────────────
  S=$(( T - 18 ))
  MODES=(noop)
  GAMES=(gbs beauty_contest)
  PLAYERS=(2 3 4)
  MODE=${MODES[$((S / 6))]}
  GAME=${GAMES[$(( (S / 3) % 2 ))]}
  N=${PLAYERS[$((S % 3))]}
  CONFIG="config/reasoning_sweep/${GAME}_${MODE}_${N}p_20b.yaml"
fi

echo "[$(date)] config=${CONFIG}"
"$PYTHON" scripts/run_episode.py --config "${CONFIG}"
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} finished."
