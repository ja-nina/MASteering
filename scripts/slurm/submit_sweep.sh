#!/bin/bash
# Submit one SLURM array per game group of a generated sweep, sized to the group: one task per
# ~EPISODES_PER_TASK episodes (runs x episodes), so big groups (werewolf, avalon) get more tasks.
# Usage: bash scripts/slurm/submit_sweep.sh config/runs/NAME [episodes per task, default 400] [extra sbatch args]
#        DRY_RUN=1 bash scripts/slurm/submit_sweep.sh ...   (print the arrays, submit nothing)
set -euo pipefail
SWEEP="${1:?Usage: submit_sweep.sh <config/runs/NAME> [episodes per task] [sbatch args]}"
PER_TASK="${2:-400}"; shift; shift || true
# "<group dir> <episodes>" per group; reads each config's top-level `episodes:` line (fast for 16k files)
"${PYTHON:-python}" - "$SWEEP" <<'EOF' |
import re, sys
from pathlib import Path
for group in sorted(p for p in Path(sys.argv[1]).iterdir() if p.is_dir()):
    total = sum(int(re.search(r"^episodes: (\d+)$", f.read_text(), re.M).group(1)) for f in group.glob("*.yaml"))
    print(group.as_posix(), total)
EOF
while read -r GROUP EPISODES; do
  EPISODES="${EPISODES//[!0-9]/}"  # drop a stray carriage return (Windows Python)
  TASKS=$(( (EPISODES + PER_TASK - 1) / PER_TASK ))
  if [ -n "${DRY_RUN:-}" ]; then JOB=dry-run; else
    JOB=$(sbatch --parsable --array=0-$((TASKS - 1)) --export=ALL,NUM_SHARDS=$TASKS "$@" scripts/slurm/run_games.sbatch "$GROUP/"); fi
  echo "$(basename "$GROUP"): $EPISODES episodes, $TASKS task(s): $JOB"
done
