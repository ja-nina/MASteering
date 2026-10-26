#!/bin/bash
#SBATCH --job-name=ma-report
#SBATCH -p gpu-a40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/report-%j.out
# Usage: sbatch scripts/slurm/report.sbatch [sweep name, default main] [sweep file] [model key]
#   another model's logs: sbatch scripts/slurm/report.sbatch main_qwen3.8-27b config/sweeps/focus.yaml qwen3.8-27b
# Summary CSV + HTML report of a sweep, on a compute node: the first pass over older episodes reads their
# JSONLs and needs more memory and time than the submit node allows. The partition only takes GPU jobs
# (and at most 10 CPUs per GPU);
# the GPU stays idle. Then, from the submit node: git add results/<sweep> && git commit && git push
# (agents.csv.gz is split into agents-2.csv.gz, ... past 90 MB: GitHub refuses files over 100 MB)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm-logs
PYTHON="${PYTHON:-$(command -v python || true)}"
[ -x "$PYTHON" ] || { echo "no python found: activate the pvf env before sbatch" >&2; exit 1; }
SWEEP="${1:-main}"
SWEEP_FILE="${2:-config/sweeps/$SWEEP.yaml}"
MODEL="${3:-}"
LOGS="${MA_LOG_ROOT:-/scratch/inf0/user/nzukowsk/MASteer/logs}/$SWEEP"
FALLBACK="${MA_FALLBACK_LOG_DIR:-/BS/ForNet/nobackup/MASteer/logs}/$SWEEP"
OUT="results/$SWEEP"
echo "$(date)  $(hostname)  job=${SLURM_JOB_ID:-local}  $LOGS (+ $FALLBACK) -> $OUT"
# scores of older episodes of the games that pay every step (side files; done once, then skipped)
"$PYTHON" scripts/backfill_rewards.py "$LOGS" "$FALLBACK" --jobs "${SLURM_CPUS_PER_TASK:-8}"
# token counts of older episodes (side files; done once, then skipped)
# (optional: if it fails, the report shows the counted episodes and '–' for the rest)
"$PYTHON" scripts/backfill_tokens.py "$LOGS" "$FALLBACK" --jobs "${SLURM_CPUS_PER_TASK:-8}" \
    || echo "token backfill incomplete: rerun to count the rest"
"$PYTHON" scripts/summarize_runs.py "$LOGS" "$FALLBACK" --out "$OUT/agents.csv.gz" --jobs "${SLURM_CPUS_PER_TASK:-8}"
# norms of the persona vectors per layer, for the push size of the random directions (report section 26)
"$PYTHON" scripts/vector_norms.py --sweep "$SWEEP_FILE" ${MODEL:+--model "$MODEL"} --out "$OUT/vector_norms.json" \
    || echo "vector norms unavailable: section 26 plots over c instead of the push size"
"$PYTHON" scripts/report.py "$OUT/agents.csv.gz" --sweep "$SWEEP_FILE" ${MODEL:+--model "$MODEL"} --logs "$LOGS" "$FALLBACK" --out "$OUT/report.html"
"$PYTHON" scripts/steering_mini_report.py "$OUT/agents.csv.gz" --logs "$LOGS" "$FALLBACK" --out "$OUT/steering_mini_report.html"
grep -o "[0-9]* episodes in this report" "$OUT/report.html"
echo "$(date)  done: git add $OUT && git commit -m 'Report: $SWEEP' && git push"
