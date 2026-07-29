#!/bin/bash
# Submit all TextArena game experiments to SLURM as a single array job.
# Run from the repo root: bash scripts/submit_all_games.sh
#
# The array encodes game × shard: task_id = game_index * 10 + shard
# 22 games × 10 shards = 220 tasks; --array=0-219%30 caps concurrency at 30.

set -euo pipefail

JOB=$(sbatch scripts/textarena/slurm/run_all_textarena_qwen.slurm | awk '{print $NF}')
echo "Submitted all TextArena baselines → job array ${JOB}"
echo "22 games × 10 shards = 220 tasks, 30 concurrent max."
echo ""
echo "Game index mapping (task_id // 10):"
echo "   0 debate               8 codenames            16 settlers_of_catan"
echo "   1 mafia                9 character_conclave   17 high_society"
echo "   2 negotiation         10 truth_and_deception  18 market_entry"
echo "   3 taboo               11 liars_dice           19 poker"
echo "   4 three_player_ipd    12 iterated_ipd         20 scorable_games"
echo "   5 three_player_gops   13 iterated_ultimatum   21 diplomacy"
echo "   6 three_player_ttt    14 used_car"
echo "   7 blind_auction       15 public_goods"
echo ""
echo "Monitor: squeue -u \$USER"
