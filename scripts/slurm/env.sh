# Shared job setup, sourced by every sbatch script here (submit from the repo root).
# Needs the PersVecFast env (vLLM 0.29 + vLLM-Hook + persvecfast + this repo's requirements):
#   micromamba activate pvf  before sbatch, or set PYTHON=/path/to/python.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm-logs
PYTHON="${PYTHON:-$(command -v python || true)}"
[ -x "$PYTHON" ] || { echo "no python found: activate the pvf env before sbatch" >&2; exit 1; }
echo "$(date)  $(hostname)  job=${SLURM_JOB_ID:-local}  python=$PYTHON"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0               # its sampler JIT-compiles with nvcc; nodes have no CUDA toolkit
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PWD/.cache/vllm}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PVF_OUT="${PVF_OUT:-/scratch/inf0/user/nzukowsk/PersVecFast/out}"
# vLLM-Hook files per job, on node-local disk (BeeGFS cannot free delivered capture rows)
export MIA_APERTURE_DIR="${MIA_APERTURE_DIR:-/tmp/ma-aperture-${SLURM_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID:-0}}"
export TESTBED_MIA_DIR="${TESTBED_MIA_DIR:-/tmp/ma-mia-${SLURM_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$MIA_APERTURE_DIR"

# /tmp watcher: every TMP_WATCH_SECONDS (default 10, 0 = off) log the node's /tmp use and its
# largest entries (size, owner) to slurm-logs/tmp-<job>.log, to see who fills node-local disk
TMP_WATCH_SECONDS="${TMP_WATCH_SECONDS:-10}"
TMP_LOG="slurm-logs/tmp-${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}.log"
watch_tmp() {
  while true; do
    {
      echo "== $(date +%T)  $(hostname)  $(df -h /tmp | awk 'NR==2 {print $3 " used, " $4 " free of " $2}')"
      du -sm /tmp/* 2>/dev/null | sort -rn | head -8 | while read -r mb path; do
        printf '%8s MB  %-10s %s\n' "$mb" "$(stat -c %U "$path" 2>/dev/null)" "$path"
      done
    } >> "$TMP_LOG"
    sleep "$TMP_WATCH_SECONDS"
  done
}
WATCH_PID=""
if [ "$TMP_WATCH_SECONDS" -gt 0 ]; then
  watch_tmp &
  WATCH_PID=$!
  echo "/tmp usage every ${TMP_WATCH_SECONDS}s -> $TMP_LOG"
fi
trap '[ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null; rm -rf "$MIA_APERTURE_DIR" "$TESTBED_MIA_DIR"' EXIT
