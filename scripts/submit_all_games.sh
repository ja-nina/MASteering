#!/bin/bash
# Submit all game experiments to SLURM.
# Run from the repo root: bash scripts/submit_all_games.sh

set -euo pipefail

echo "Submitting all game experiments..."

WEREWOLF=$(sbatch scripts/werewolf/slurm/run_werewolf_qwen.slurm  | awk '{print $NF}')
AVALON=$(  sbatch scripts/avalon/slurm/run_avalon_qwen.slurm       | awk '{print $NF}')
HANABI=$(  sbatch scripts/hanabi/slurm/run_hanabi_qwen.slurm       | awk '{print $NF}')
OVERCOOKED=$(sbatch scripts/overcooked/slurm/run_overcooked_qwen.slurm | awk '{print $NF}')
DIPLOMACY=$(sbatch scripts/diplomacy/slurm/run_diplomacy_qwen.slurm | awk '{print $NF}')

echo "  Werewolf   → job ${WEREWOLF}  (100 tasks, players 6-15)"
echo "  Avalon     → job ${AVALON}    (60 tasks, players 5-10)"
echo "  Hanabi     → job ${HANABI}    (40 tasks, players 2-5)"
echo "  Overcooked → job ${OVERCOOKED} (30 tasks, 3 layouts)"
echo "  Diplomacy  → job ${DIPLOMACY}  (10 tasks, TextArena)"
echo ""
echo "Monitor with: squeue -u \$USER"
