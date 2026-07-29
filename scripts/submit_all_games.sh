#!/bin/bash
# Submit all TextArena game experiments to SLURM as a single array job.
# Run from the repo root: bash scripts/submit_all_games.sh
#
# The array encodes game × shard: task_id = game_index * 10 + shard
# 22 games × 10 shards = 220 tasks; --array=0-219%30 caps concurrency at 30.

set -euo pipefail

JOB=$(sbatch scripts/textarena/slurm/run_all_textarena_qwen.slurm | awk '{print $NF}')
echo "Submitted all TextArena baselines → job array ${JOB}"
echo "18 games × 10 shards = 180 tasks, 30 concurrent max."
echo ""
echo "Game index mapping (task_id // 10):"
echo "   0 debate               6 codenames            12 settlers_of_catan"
echo "   1 mafia                7 truth_and_deception  13 high_society"
echo "   2 negotiation          8 liars_dice           14 market_entry"
echo "   3 three_player_ipd     9 iterated_ipd         15 poker"
echo "   4 three_player_gops   10 iterated_ultimatum   16 scorable_games"
echo "   5 three_player_ttt    11 public_goods         17 diplomacy"
echo ""
echo "Monitor: squeue -u \$USER"
