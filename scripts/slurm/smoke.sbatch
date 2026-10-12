#!/bin/bash
#SBATCH --job-name=ma-smoke-engines
#SBATCH -p gpu-a40
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-logs/smoke-engines-%j.out
# GPU checks for the vLLM-Hook engines: two live engines, per-request steering (vector and prompt
# persona) moves the monitor, coefficient 0 == unsteered, image prompts. Samples are saved to
# results/smoke/engines/<model>/.   Usage: sbatch scripts/slurm/smoke.sbatch [model]
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/env.sh"
MODEL="${1:-qwen3.5-9b}"
"$PYTHON" scripts/smoke_engines.py --model "$MODEL" --out "results/smoke/engines/$MODEL"
