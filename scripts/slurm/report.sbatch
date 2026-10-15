#!/bin/bash
#SBATCH --job-name=ma-report
#SBATCH -p gpu-a40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-logs/report-%j.out
# Usage: sbatch scripts/slurm/report.sbatch [sweep name, default main]
# Summary CSV + HTML report of a sweep, on a compute node: the first pass over older episodes reads their
# JSONLs and needs more memory and time than the submit node allows. The partition only takes GPU jobs
# (and at most 10 CPUs per GPU);
# the GPU stays idle. Then, from the submit node: git add results/<sweep> && git commit && git push
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm-logs
PYTHON="${PYTHON:-$(command -v python || true)}"
[ -x "$PYTHON" ] || { echo "no python found: activate the pvf env before sbatch" >&2; exit 1; }
SWEEP="${1:-main}"
LOGS="${MA_LOG_ROOT:-/scratch/inf0/user/nzukowsk/MASteer/logs}/$SWEEP"
FALLBACK="${MA_FALLBACK_LOG_DIR:-/BS/ForNet/nobackup/MASteer/logs}/$SWEEP"
OUT="results/$SWEEP"
echo "$(date)  $(hostname)  job=${SLURM_JOB_ID:-local}  $LOGS (+ $FALLBACK) -> $OUT"
"$PYTHON" scripts/summarize_runs.py "$LOGS" "$FALLBACK" --out "$OUT/agents.csv.gz" --jobs "${SLURM_CPUS_PER_TASK:-8}"
"$PYTHON" scripts/report.py "$OUT/agents.csv.gz" --sweep "config/sweeps/$SWEEP.yaml" --logs "$LOGS" --out "$OUT/report.html"
grep -o "[0-9]* episodes in this report" "$OUT/report.html"
echo "$(date)  done: git add $OUT/agents.csv.gz $OUT/report.html && git commit -m 'Report: $SWEEP' && git push"
