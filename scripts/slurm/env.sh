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
trap 'rm -rf "$MIA_APERTURE_DIR" "$TESTBED_MIA_DIR"' EXIT
