#!/bin/bash
#SBATCH --job-name=ma-smoke-games
#SBATCH -p gpu-a40
#SBATCH -c 16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-logs/smoke-games-%j.out
# Smoke test of every game x persona channel (config/sweeps/smoke.yaml) in ONE job, then the
# samples page: results/smoke/samples.html (+ .csv status per run). A failing game only fails
# its own episodes; the page lists which runs failed. Rerunning resumes.
# Usage: sbatch scripts/slurm/smoke_games.sbatch [sweep file, default config/sweeps/smoke.yaml]
source "${SLURM_SUBMIT_DIR:-.}/scripts/slurm/env.sh"
SWEEP="${1:-config/sweeps/smoke.yaml}"
read -r NAME LOGS < <("$PYTHON" -c "import sys, yaml; s = yaml.safe_load(open(sys.argv[1])); print(s['name'], s.get('logging_dir', 'logs') + '/' + s['name'])" "$SWEEP")
"$PYTHON" scripts/make_configs.py "$SWEEP"
STATUS=0
"$PYTHON" scripts/run_games.py --config-dir "config/runs/$NAME" || STATUS=$?
"$PYTHON" scripts/collect_samples.py "$LOGS" --configs "config/runs/$NAME" --out "results/$NAME/samples.html"
"$PYTHON" scripts/summarize_runs.py "$LOGS" --out "results/$NAME/agents.csv" || true
echo "run_games exit status: $STATUS (1 = some episodes failed; see results/$NAME/samples.html)"
exit $STATUS
