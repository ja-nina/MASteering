#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-17%9
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/reasoning_sweep_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/reasoning_sweep_%A_%a.err
#SBATCH -t 2:00:00
#
# Reasoning mode sweep: 3 modes x 2 games x 3 player counts = 18 tasks
#
#   mode:    noop (baseline) | non_thinking (prompted) | thinking (native CoT)
#   game:    beauty_contest | gbs
#   players: 2 | 3 | 4
#
# Prerequisites:
#   python scripts/gen_reasoning_sweep_configs.py
#
# Submit all:
#   sbatch scripts/run_reasoning_sweep.slurm
#
# Single task (e.g. gbs / thinking / 4 players = task 17):
#   sbatch --array=17 scripts/run_reasoning_sweep.slurm

set -euo pipefail
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on ${HOSTNAME}"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

MODES=(noop non_thinking thinking)
GAMES=(beauty_contest gbs)
PLAYERS=(2 3 4)

T=${SLURM_ARRAY_TASK_ID}
MODE=${MODES[$((T / 6))]}
GAME=${GAMES[$(( (T / 3) % 2 ))]}
N=${PLAYERS[$((T % 3))]}

CONFIG="config/reasoning_sweep/${GAME}_${MODE}_${N}p.yaml"

echo "[$(date)] mode=${MODE}  game=${GAME}  players=${N}  config=${CONFIG}"
"$PYTHON" scripts/run_episode.py --config "${CONFIG}"
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} finished."
