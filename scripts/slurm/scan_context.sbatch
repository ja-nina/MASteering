#!/bin/bash
#SBATCH --job-name=ma-ctxscan
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/ctxscan-%j.out
# Usage: sbatch scripts/slurm/scan_context.sbatch [sweep names, default: main main_qwen3.5-2b main_qwen3.8-27b]
# Which runs hit the context limit (prompt_too_long) and the longest prompt per run (scripts/scan_context.py),
# over the log root and the fallback root. Writes results/context_scan.csv; the GPU stays idle.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm-logs results
PYTHON="${PYTHON:-$(command -v python || true)}"
[ -x "$PYTHON" ] || { echo "no python found: activate the pvf env before sbatch" >&2; exit 1; }
ROOT="${MA_LOG_ROOT:-/scratch/inf0/user/nzukowsk/MASteer/logs}"
FALLBACK="${MA_FALLBACK_LOG_DIR:-/BS/ForNet/nobackup/MASteer/logs}"
DIRS=()
for s in ${@:-main main_qwen3.5-2b main_qwen3.8-27b}; do
  for d in "$ROOT/$s" "$FALLBACK/$s"; do [ -d "$d" ] && DIRS+=("$d"); done
done
echo "$(date)  $(hostname)  ${DIRS[*]}"
"$PYTHON" scripts/scan_context.py "${DIRS[@]}" --workers "${SLURM_CPUS_PER_TASK:-8}" --out results/context_scan.csv
echo "$(date)  done"
