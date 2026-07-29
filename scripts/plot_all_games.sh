#!/usr/bin/env bash
# Run all per-game plotting scripts. Assumes logs/ at repo root.
# Usage:  bash scripts/plot_all_games.sh [log_root] [out_root]
#
# Example on cluster:
#   bash scripts/plot_all_games.sh /scratch/inf0/user/nzukowsk/logs plots

set -euo pipefail
LOG="${1:-logs}"
OUT="${2:-plots}"

PYTHON="${PYTHON:-python}"

echo "=== Plotting Werewolf ==="
$PYTHON scripts/werewolf/plot_werewolf_results.py \
    --log-dir "${LOG}/werewolf" \
    --out     "${OUT}/werewolf"

echo "=== Plotting Avalon ==="
$PYTHON scripts/avalon/plot_avalon_results.py \
    --log-dir "${LOG}/avalon" \
    --out     "${OUT}/avalon"

echo "=== Plotting Hanabi ==="
$PYTHON scripts/hanabi/plot_hanabi_results.py \
    --log-dir "${LOG}/hanabi" \
    --out     "${OUT}/hanabi"

echo "=== Plotting Overcooked ==="
$PYTHON scripts/overcooked/plot_overcooked_results.py \
    --log-dir "${LOG}/overcooked" \
    --out     "${OUT}/overcooked"

echo "=== Plotting Mafia ==="
$PYTHON scripts/mafia/plot_mafia_results.py \
    --log-dir "${LOG}/mafia" \
    --out     "${OUT}/mafia"

echo "=== Plotting Debate ==="
$PYTHON scripts/debate/plot_debate_results.py \
    --log-dir "${LOG}/debate" \
    --out     "${OUT}/debate"

echo "=== Plotting Negotiation ==="
$PYTHON scripts/textarena/plot_negotiation_results.py \
    --log-dir "${LOG}/negotiation" \
    --out     "${OUT}/negotiation"

echo "=== Plotting Taboo ==="
$PYTHON scripts/textarena/plot_taboo_results.py \
    --log-dir "${LOG}/taboo" \
    --out     "${OUT}/taboo"

echo "=== Plotting Three-Player IPD ==="
$PYTHON scripts/textarena/plot_three_player_ipd_results.py \
    --log-dir "${LOG}/three_player_ipd" \
    --out     "${OUT}/three_player_ipd"

echo "=== Plotting Three-Player GOPS ==="
$PYTHON scripts/textarena/plot_three_player_gops_results.py \
    --log-dir "${LOG}/three_player_gops" \
    --out     "${OUT}/three_player_gops"

echo "=== Plotting Three-Player TicTacToe ==="
$PYTHON scripts/textarena/plot_three_player_tictactoe_results.py \
    --log-dir "${LOG}/three_player_tictactoe" \
    --out     "${OUT}/three_player_tictactoe"

echo "=== Plotting Blind Auction ==="
$PYTHON scripts/textarena/plot_blind_auction_results.py \
    --log-dir "${LOG}/blind_auction" \
    --out     "${OUT}/blind_auction"

echo "=== Plotting Codenames ==="
$PYTHON scripts/textarena/plot_codenames_results.py \
    --log-dir "${LOG}/codenames" \
    --out     "${OUT}/codenames"

echo "=== Plotting Character Conclave ==="
$PYTHON scripts/textarena/plot_character_conclave_results.py \
    --log-dir "${LOG}/character_conclave" \
    --out     "${OUT}/character_conclave"

echo "=== Plotting Truth and Deception ==="
$PYTHON scripts/textarena/plot_truth_and_deception_results.py \
    --log-dir "${LOG}/truth_and_deception" \
    --out     "${OUT}/truth_and_deception"

echo "=== Plotting Liar's Dice ==="
$PYTHON scripts/textarena/plot_liars_dice_results.py \
    --log-dir "${LOG}/liars_dice" \
    --out     "${OUT}/liars_dice"

echo "=== Plotting Iterated Prisoner's Dilemma ==="
$PYTHON scripts/textarena/plot_iterated_ipd_results.py \
    --log-dir "${LOG}/iterated_ipd" \
    --out     "${OUT}/iterated_ipd"

echo "=== Plotting Iterated Ultimatum Game ==="
$PYTHON scripts/textarena/plot_iterated_ultimatum_results.py \
    --log-dir "${LOG}/iterated_ultimatum" \
    --out     "${OUT}/iterated_ultimatum"

echo "=== Plotting Used Car Negotiation ==="
$PYTHON scripts/textarena/plot_used_car_results.py \
    --log-dir "${LOG}/used_car" \
    --out     "${OUT}/used_car"

echo "=== Plotting Public Goods Game ==="
$PYTHON scripts/textarena/plot_public_goods_results.py \
    --log-dir "${LOG}/public_goods" \
    --out     "${OUT}/public_goods"

echo "=== Plotting Settlers of Catan ==="
$PYTHON scripts/textarena/plot_settlers_of_catan_results.py \
    --log-dir "${LOG}/settlers_of_catan" \
    --out     "${OUT}/settlers_of_catan"

echo "=== Plotting High Society ==="
$PYTHON scripts/textarena/plot_high_society_results.py \
    --log-dir "${LOG}/high_society" \
    --out     "${OUT}/high_society"

echo "=== Plotting Market Entry Game ==="
$PYTHON scripts/textarena/plot_market_entry_results.py \
    --log-dir "${LOG}/market_entry" \
    --out     "${OUT}/market_entry"

echo "=== Plotting Poker ==="
$PYTHON scripts/textarena/plot_poker_results.py \
    --log-dir "${LOG}/poker" \
    --out     "${OUT}/poker"

echo "=== Plotting Scorable Games ==="
$PYTHON scripts/textarena/plot_scorable_games_results.py \
    --log-dir "${LOG}/scorable_games" \
    --out     "${OUT}/scorable_games"

echo "=== Plotting GBS ==="
$PYTHON scripts/gbs/plot_gbs_results.py \
    --log-dir "${LOG}" \
    --out     "${OUT}/gbs"

echo ""
echo "All done. Plots in ${OUT}/"
