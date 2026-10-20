#!/bin/bash
# Submit one SLURM array per game group of a generated sweep, sized to the group: one task per
# ~EPISODES_PER_TASK episodes (runs x episodes), so big groups (werewolf, avalon) get more tasks.
# Usage: bash scripts/slurm/submit_sweep.sh config/runs/NAME [episodes per task, default 400] [extra sbatch args]
#        DRY_RUN=1 bash scripts/slurm/submit_sweep.sh ...   (print the arrays, submit nothing)
#        ONLY='avalon|werewolf' bash scripts/slurm/submit_sweep.sh ...   (only game groups matching this regex)
#        PART=1/2 bash scripts/slurm/submit_sweep.sh ...    (first half of the game groups; 2/2 = the rest)
#        THROTTLE=10 bash scripts/slurm/submit_sweep.sh ... (at most 10 running tasks per array: --array=0-N%10)
set -euo pipefail
SWEEP="${1:?Usage: submit_sweep.sh <config/runs/NAME> [episodes per task] [sbatch args]}"
PER_TASK="${2:-400}"; shift; shift || true
# "<group dir> <episodes>" per group; reads each config's top-level `episodes:` line (fast for 16k files)
"${PYTHON:-python}" - "$SWEEP" <<'EOF' |
import re, sys
from pathlib import Path
import os
groups = sorted(p for p in Path(sys.argv[1]).iterdir() if p.is_dir())
groups = [g for g in groups if re.search(os.environ.get("ONLY") or "", g.name)]
part, parts = map(int, (os.environ.get("PART") or "1/1").split("/"))
size = -(-len(groups) // parts)
for group in groups[(part - 1) * size: part * size]:
    total = sum(int(re.search(r"^episodes: (\d+)$", f.read_text(), re.M).group(1)) for f in group.glob("*.yaml"))
    print(group.as_posix(), total)
EOF
while read -r GROUP EPISODES; do
  EPISODES="${EPISODES//[!0-9]/}"  # drop a stray carriage return (Windows Python)
  TASKS=$(( (EPISODES + PER_TASK - 1) / PER_TASK ))
  if [ -n "${DRY_RUN:-}" ]; then JOB=dry-run; else
    JOB=$(sbatch --parsable --array=0-$((TASKS - 1))${THROTTLE:+%$THROTTLE} --job-name "$(basename "$GROUP")" "$@" scripts/slurm/run_games.sbatch "$GROUP/" --num-shards "$TASKS"); fi
  echo "$(basename "$GROUP"): $EPISODES episodes, $TASKS task(s)${THROTTLE:+, at most $THROTTLE at once}: $JOB"
done
