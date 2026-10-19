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
# episodes go here (<this>/<sweep>/<game group>/<run_id>) while the sweep's logging_dir (scratch) fails;
# resume and summarize_runs.py look in both. MA_FALLBACK_LOG_DIR= (empty) turns it off
MA_FALLBACK_LOG_DIR="${MA_FALLBACK_LOG_DIR-/BS/ForNet/nobackup/MASteer/logs}"
if [ -n "$MA_FALLBACK_LOG_DIR" ] && timeout 30 mkdir -p "$MA_FALLBACK_LOG_DIR" 2>/dev/null; then
  export MA_FALLBACK_LOG_DIR
  echo "fallback logs -> $MA_FALLBACK_LOG_DIR"
else
  echo "no fallback log dir (${MA_FALLBACK_LOG_DIR:-off})"; unset MA_FALLBACK_LOG_DIR
fi
# vLLM-Hook's captured activations per job (written, read back, delivered per request: the monitor's
# bottleneck). In RAM (/dev/shm) when it has room, else node-local disk; never BeeGFS (cannot free rows).
# MA_APERTURE_ROOT overrides; a task needs ~6-8 GB there.
if [ -z "${MA_APERTURE_ROOT:-}" ]; then
  MA_APERTURE_ROOT=/tmp
  if [ -d /dev/shm ] && [ -w /dev/shm ] && [ "$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc 0-9)" -ge 40 ] 2>/dev/null; then
    MA_APERTURE_ROOT=/dev/shm
  fi
fi
export MIA_APERTURE_DIR="${MIA_APERTURE_DIR:-$MA_APERTURE_ROOT/ma-aperture-${SLURM_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID:-0}}"
echo "capture files (aperture) -> $MIA_APERTURE_DIR"
export TESTBED_MIA_DIR="${TESTBED_MIA_DIR:-/tmp/ma-mia-${SLURM_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$MIA_APERTURE_DIR"
# capture engine's GPU buffer for captured rows: 8 GiB = 32768 rows x 32 layers x 8 KB (default 4 GiB,
# 16384 rows, filled up during a long prefill while the previous batch was still being written out).
# It lives outside vLLM's share, so the capture engine runs at memory 0.75 (0.85 leaves only 6.7 GiB)
export MIA_APERTURE_GPU_BYTES="${MIA_APERTURE_GPU_BYTES:-8589934592}"
# how long a capture read waits for its activations (vLLM-Hook's default 60 s; a busy node's /tmp can be slower).
# A read that still times out is retried once, then those turns are logged without monitor scores
export MIA_DELIVERY_TIMEOUT_S="${MIA_DELIVERY_TIMEOUT_S:-300}"

# /tmp watcher: every TMP_WATCH_SECONDS (default 10, 0 = off) log the node's /tmp use and its
# largest entries (size, owner) to slurm-logs/ (TMP_LOG), to see who fills node-local disk
TMP_WATCH_SECONDS="${TMP_WATCH_SECONDS:-10}"
# named like SLURM's own logs: tmp-<job>.log, or tmp-<array job>_<task>.log for an array task
TMP_LOG="slurm-logs/tmp-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}.log"
watch_tmp() {
  set +e +o pipefail  # du cannot read other users' files: that must not end the watcher (it inherits set -e)
  while true; do
    {
      # df counts deleted-but-open files and unreadable dirs that du cannot see: the gap between them
      [ "$MA_APERTURE_ROOT" != /tmp ] && echo "== $(date +%T)  $MA_APERTURE_ROOT: $(df -h "$MA_APERTURE_ROOT" | awk 'NR==2 {print $3 " used, " $4 " free"}'), aperture $(timeout 5 du -sm "$MIA_APERTURE_DIR" 2>/dev/null | cut -f1) MB"
      echo "== $(date +%T)  $(hostname)  $(df -h /tmp | awk 'NR==2 {print $3 " used, " $4 " free of " $2}'),"           "$(timeout 5 du -sm /tmp 2>/dev/null | cut -f1) MB visible to du"
      timeout 5 du -sm /tmp/* /tmp/.[!.]* 2>/dev/null | sort -rn | head -8 | while read -r mb path; do
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
# the cleanup must not fail: under set -e a failing command here (kill of a watcher that already ended) becomes the
# job's exit code, and a task that finished cleanly shows up as FAILED in sacct
trap '{ [ -z "$WATCH_PID" ] || kill "$WATCH_PID" 2>/dev/null || true; rm -rf "$MIA_APERTURE_DIR" "$TESTBED_MIA_DIR"; } || true' EXIT
trap 'exit 143' TERM INT  # scancel / time limit: still run the cleanup above (files in /dev/shm hold RAM)
