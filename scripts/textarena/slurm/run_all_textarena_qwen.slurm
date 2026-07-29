#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-219%30
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/textarena_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/textarena_%A_%a.err
#SBATCH -t 08:00:00
#
# All TextArena noop baselines — single unified array.
# Encoding: task_id = game_index * NUM_SHARDS + shard
# 22 games × 10 shards = 220 tasks; 30 concurrent max (%30).
#
# Game index → config mapping (update CONFIGS array when adding games):
#   0  debate               8  codenames            16  settlers_of_catan
#   1  mafia                9  character_conclave   17  high_society
#   2  negotiation         10  truth_and_deception  18  market_entry
#   3  taboo               11  liars_dice           19  poker
#   4  three_player_ipd    12  iterated_ipd         20  scorable_games
#   5  three_player_gops   13  iterated_ultimatum   21  diplomacy
#   6  three_player_ttt    14  used_car
#   7  blind_auction       15  public_goods
#
# Submit:
#   sbatch scripts/textarena/slurm/run_all_textarena_qwen.slurm

set -euo pipefail

NUM_SHARDS=10
RESUME_HOP_LIMIT=20
HOP=${1:-0}

GAME_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))
SHARD=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))

CONFIGS=(
    "config/games/debate/debate_noop_2p.yaml"                                    # 0
    "config/games/mafia/mafia_noop_8p.yaml"                                      # 1
    "config/games/negotiation/negotiation_noop_2p.yaml"                          # 2
    "config/games/taboo/taboo_noop_2p.yaml"                                      # 3
    "config/games/three_player_ipd/three_player_ipd_noop_3p.yaml"                # 4
    "config/games/three_player_gops/three_player_gops_noop_3p.yaml"              # 5
    "config/games/three_player_tictactoe/three_player_tictactoe_noop_3p.yaml"    # 6
    "config/games/blind_auction/blind_auction_noop_4p.yaml"                      # 7
    "config/games/codenames/codenames_noop_4p.yaml"                              # 8
    "config/games/character_conclave/character_conclave_noop_4p.yaml"            # 9
    "config/games/two_rooms/truth_and_deception_noop_2p.yaml"                    # 10
    "config/games/liars_dice/liars_dice_noop_3p.yaml"                            # 11
    "config/games/iterated_ipd/iterated_ipd_noop_2p.yaml"                        # 12
    "config/games/iterated_ultimatum/iterated_ultimatum_noop_2p.yaml"             # 13
    "config/games/used_car/used_car_noop_2p.yaml"                                # 14
    "config/games/public_goods/public_goods_noop_4p.yaml"                        # 15
    "config/games/settlers_of_catan/settlers_of_catan_noop_4p.yaml"              # 16
    "config/games/high_society/high_society_noop_4p.yaml"                        # 17
    "config/games/market_entry/market_entry_noop_4p.yaml"                        # 18
    "config/games/poker/poker_noop_4p.yaml"                                      # 19
    "config/games/scorable_games/scorable_games_noop_4p.yaml"                    # 20
    "config/games/diplomacy/diplomacy_noop_5p.yaml"                              # 21
)

CONFIG="${CONFIGS[$GAME_IDX]}"
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} → game ${GAME_IDX} shard ${SHARD} config=${CONFIG} (hop ${HOP})"

PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"
mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

"$PYTHON" scripts/run_episode.py --config "${CONFIG}" \
          --shard "${SHARD}" --num-shards "${NUM_SHARDS}"
echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} finished this chunk."

REMAINING=$("$PYTHON" - "${CONFIG}" "${SHARD}" "${NUM_SHARDS}" <<'PYEOF'
import glob, re, sys, yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
shard, num_shards = int(sys.argv[2]), int(sys.argv[3])
run_dir = f"{cfg['logging']['dir'].rstrip('/')}/{cfg['run_id']}"

done = set()
for p in glob.glob(f"{run_dir}/episode_*.summary.json"):
    m = re.search(r"episode_(\d+)\.summary\.json$", p)
    if m:
        done.add(int(m.group(1)))

assigned = [ep for ep in range(cfg["episodes"]) if ep % num_shards == shard]
print(sum(1 for ep in assigned if ep not in done))
PYEOF
)

if [ "${REMAINING}" -gt 0 ] && [ "${HOP}" -lt "${RESUME_HOP_LIMIT}" ]; then
    echo "[$(date)] ${REMAINING} episode(s) remaining — requeuing task ${SLURM_ARRAY_TASK_ID} (hop $((HOP + 1)))"
    sbatch --dependency=afterany:"${SLURM_JOB_ID}" \
           --array="${SLURM_ARRAY_TASK_ID}" \
           "$0" $((HOP + 1))
elif [ "${REMAINING}" -gt 0 ]; then
    echo "[$(date)] WARNING: hop limit (${RESUME_HOP_LIMIT}) reached with ${REMAINING} remaining for task ${SLURM_ARRAY_TASK_ID}."
else
    echo "[$(date)] All episodes complete for game ${GAME_IDX} shard ${SHARD}."
fi
