#!/bin/bash
#SBATCH -p gpu16                          # FILL_IN: partition name on your cluster
#SBATCH -c 10                             # episode generation is CPU-bound; model fwd pass is GPU-bound
#SBATCH --gres=gpu:1                      # FILL_IN: e.g. gpu:a100:1 if your cluster requires a specific type
#SBATCH -t 48:00:00                       # FILL_IN: see walltime note below
#SBATCH -o /scratch/inf0/user/nzukowsk/logs/layer_sweep_%j.out
#SBATCH -e /scratch/inf0/user/nzukowsk/logs/layer_sweep_%j.err
#
# Runs the disk-bounded full-layer SAE sweep (resid + MLP, layers 10..last)
# for one game via scripts/run_layer_sweep.py. Submit once per game, from the
# repo root:
#
#   sbatch scripts/run_layer_sweep.slurm beauty_contest
#   sbatch scripts/run_layer_sweep.slurm gbs
#
# Not a job array: run_layer_sweep.py already loops over every layer
# internally (see docs/superpowers/specs/2026-06-16-qwen3-4b-layer-sweep-design.md).
#
# WALLTIME NOTE: a transformer forward pass computes every layer regardless
# of which one is hooked, so this re-runs the forward pass once per swept
# layer instead of once total (~26x for layers 10-35 on Qwen3-4B) — that's
# the deliberate trade-off for keeping disk usage bounded (see design doc
# section 4). Before trusting the -t value above for a full sweep, time one
# layer first:
#   sbatch --time=2:00:00 --wrap='micromamba run -n ma-testbed python scripts/run_layer_sweep.py --game beauty_contest --start-layer 10 --end-layer 10'
# and extrapolate to however many layers you're sweeping.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "[$(date)] Job ${SLURM_JOB_ID} starting on ${HOSTNAME}"

GAME="${1:-beauty_contest}"               # beauty_contest | gbs

mkdir -p /scratch/inf0/user/nzukowsk/logs

micromamba run -n ma-testbed python scripts/run_layer_sweep.py \
  --game "${GAME}" \
  --model Qwen/Qwen3-4B \
  --start-layer 10 \
  --num-episodes 500 \
  --max-rounds 4 \
  --num-players 4 \
  --activations-dir /scratch/inf0/user/nzukowsk/activations \
  --sae-dir /scratch/inf0/user/nzukowsk/sae \
  --vectors-dir vectors

echo "[$(date)] Job ${SLURM_JOB_ID} finished."
