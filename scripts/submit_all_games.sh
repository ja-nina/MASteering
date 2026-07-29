#!/bin/bash
# Submit all game experiments to SLURM.
# Run from the repo root: bash scripts/submit_all_games.sh

set -euo pipefail

echo "Submitting all game experiments..."

WEREWOLF=$(      sbatch scripts/werewolf/slurm/run_werewolf_qwen.slurm             | awk '{print $NF}')
AVALON=$(        sbatch scripts/avalon/slurm/run_avalon_qwen.slurm                 | awk '{print $NF}')
HANABI=$(        sbatch scripts/hanabi/slurm/run_hanabi_qwen.slurm                 | awk '{print $NF}')
OVERCOOKED=$(    sbatch scripts/overcooked/slurm/run_overcooked_qwen.slurm         | awk '{print $NF}')
MAFIA=$(         sbatch scripts/mafia/slurm/run_mafia_qwen.slurm                   | awk '{print $NF}')
DEBATE=$(        sbatch scripts/debate/slurm/run_debate_qwen.slurm                 | awk '{print $NF}')
NEGOTIATION=$(   sbatch scripts/textarena/slurm/run_negotiation_qwen.slurm         | awk '{print $NF}')
TABOO=$(         sbatch scripts/textarena/slurm/run_taboo_qwen.slurm               | awk '{print $NF}')
TPIPD=$(         sbatch scripts/textarena/slurm/run_three_player_ipd_qwen.slurm    | awk '{print $NF}')
TPGOPS=$(        sbatch scripts/textarena/slurm/run_three_player_gops_qwen.slurm   | awk '{print $NF}')
TPTTT=$(         sbatch scripts/textarena/slurm/run_three_player_tictactoe_qwen.slurm | awk '{print $NF}')
BLIND_AUCTION=$( sbatch scripts/textarena/slurm/run_blind_auction_qwen.slurm       | awk '{print $NF}')
CODENAMES=$(     sbatch scripts/textarena/slurm/run_codenames_qwen.slurm           | awk '{print $NF}')
CONCLAVE=$(      sbatch scripts/textarena/slurm/run_character_conclave_qwen.slurm  | awk '{print $NF}')
TRUTH=$(         sbatch scripts/textarena/slurm/run_truth_and_deception_qwen.slurm | awk '{print $NF}')
LIARS_DICE=$(    sbatch scripts/textarena/slurm/run_liars_dice_qwen.slurm          | awk '{print $NF}')
ITER_IPD=$(      sbatch scripts/textarena/slurm/run_iterated_ipd_qwen.slurm        | awk '{print $NF}')
ITER_ULT=$(      sbatch scripts/textarena/slurm/run_iterated_ultimatum_qwen.slurm  | awk '{print $NF}')
USED_CAR=$(      sbatch scripts/textarena/slurm/run_used_car_qwen.slurm            | awk '{print $NF}')
PUBLIC_GOODS=$(  sbatch scripts/textarena/slurm/run_public_goods_qwen.slurm       | awk '{print $NF}')
SETTLERS=$(      sbatch scripts/textarena/slurm/run_settlers_of_catan_qwen.slurm  | awk '{print $NF}')
HIGH_SOCIETY=$(  sbatch scripts/textarena/slurm/run_high_society_qwen.slurm       | awk '{print $NF}')
MARKET_ENTRY=$(  sbatch scripts/textarena/slurm/run_market_entry_qwen.slurm       | awk '{print $NF}')
POKER=$(         sbatch scripts/textarena/slurm/run_poker_qwen.slurm              | awk '{print $NF}')
SCORABLE=$(      sbatch scripts/textarena/slurm/run_scorable_games_qwen.slurm     | awk '{print $NF}')

echo "  Werewolf            → job ${WEREWOLF}       (players 6-15, symbolic)"
echo "  Avalon              → job ${AVALON}         (players 5-10, symbolic)"
echo "  Hanabi              → job ${HANABI}         (players 2-5, symbolic)"
echo "  Overcooked          → job ${OVERCOOKED}     (3 layouts, symbolic)"
echo "  SecretMafia         → job ${MAFIA}          (8p, TextArena SecretMafia-v0)"
echo "  Debate              → job ${DEBATE}         (2p, TextArena Debate-v0)"
echo "  Negotiation         → job ${NEGOTIATION}    (2p, TextArena SimpleNegotiation-v0)"
echo "  Taboo               → job ${TABOO}          (2p, TextArena Taboo-v0)"
echo "  ThreePlayerIPD      → job ${TPIPD}          (3p, TextArena ThreePlayerIPD-v0)"
echo "  ThreePlayerGOPS     → job ${TPGOPS}         (3p, TextArena ThreePlayerGOPS-v0)"
echo "  ThreePlayerTicTacToe→ job ${TPTTT}          (3p, TextArena ThreePlayerTicTacToe-v0)"
echo "  BlindAuction        → job ${BLIND_AUCTION}  (4p, TextArena SimpleBlindAuction-v0)"
echo "  Codenames           → job ${CODENAMES}      (4p, TextArena Codenames-v0)"
echo "  CharacterConclave   → job ${CONCLAVE}       (4p, TextArena CharacterConclave-v0)"
echo "  TruthAndDeception   → job ${TRUTH}          (2p, TextArena TruthAndDeception-v0)"
echo "  LiarsDice           → job ${LIARS_DICE}     (3p, TextArena LiarsDice-v0)"
echo "  IteratedIPD         → job ${ITER_IPD}       (2p, TextArena IteratedPrisonersDilemma-v0)"
echo "  IteratedUltimatum   → job ${ITER_ULT}       (2p, TextArena IteratedUltimatumGame-v0)"
echo "  UsedCarNegotiation  → job ${USED_CAR}       (2p, TextArena UsedCarNegotiation-v0)"
echo "  PublicGoods         → job ${PUBLIC_GOODS}   (4p, TextArena PublicGoodsGame-v0)"
echo "  SettlersOfCatan     → job ${SETTLERS}       (4p, TextArena SettlersOfCatan-v0)"
echo "  HighSociety         → job ${HIGH_SOCIETY}   (4p, TextArena HighSociety-v0)"
echo "  MarketEntry         → job ${MARKET_ENTRY}   (4p, TextArena MarketEntryGame-v0)"
echo "  Poker               → job ${POKER}          (4p, TextArena Poker-v0)"
echo "  ScorableGames       → job ${SCORABLE}       (4p, TextArena ScorableGames-v0)"
echo ""
echo "Monitor with: squeue -u \$USER"
