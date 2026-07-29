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

echo "=== Plotting Diplomacy ==="
$PYTHON scripts/diplomacy/plot_diplomacy_results.py \
    --log-dir "${LOG}/diplomacy" \
    --out     "${OUT}/diplomacy"

echo "=== Plotting GBS ==="
$PYTHON scripts/gbs/plot_gbs_results.py \
    --log-dir "${LOG}" \
    --out     "${OUT}/gbs"

echo ""
echo "All done. Plots in ${OUT}/"
