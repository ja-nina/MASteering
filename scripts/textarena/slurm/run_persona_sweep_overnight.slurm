#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --array=0-79%16
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/sweep_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/sweep_%A_%a.err
#SBATCH -t 04:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  Persona vector sweep — 12 traits × 2 games (overnight run)
#  coefficient: 1.25 (adaptive mode)  |  controls: mafia & debate noop
# ════════════════════════════════════════════════════════════════════════════
#
# Encoding: task_id = config_index * NUM_SHARDS + shard
#   14 configs × 5 shards = 70 tasks
#   %14 concurrent → one active shard per config at a time
#
# Time estimate per shard:
#   Mafia  (20 eps / 5 shards = 4 eps)  ≈ 60 min
#   Debate (50 eps / 5 shards = 10 eps) ≈ 30 min
#   → total wall time ≈ 5 rounds × 1 h = ~5 h
#
# Configs
# ───────────────────────────────────────────────────────────────────────────
# idx  game    trait                   hypothesis
#  0   mafia   evil                    more overt tells vs opportunistic?
#  1   mafia   trustworthiness         paradox: trustworthy wolf survives longer?
#  2   mafia   charismatic             charm buys immunity from votes?
#  3   mafia   sycophantic             over-agreeing: camouflage or red flag?
#  4   mafia   loquacious              verbosity → more tells?
#  5   mafia   ethical [POSITIVE]      ethics-driven wolf refuses to deceive?
#  6   debate  charismatic             charm wins arguments? (cross-game pair with idx 2)
#  7   debate  sycophantic             capitulation condition
#  8   debate  epistemic-humility      self-doubt weakens adversarial stance?
#  9   debate  sarcastic               confrontation helps or hurts?
# 10   debate  goal-oriented           outcome focus sharpens argumentation?
# 11   debate  ethical [POSITIVE]      moral integrity: credibility boost or pulling punches?
# 12   mafia   [CONTROL noop+probe]    re-run control (old format, no chunks)
# 13   debate  [CONTROL noop+probe]    re-run control (old format, no chunks)
# 14   mafia   [CONTROL noop+chunks]   fresh noop with chunk time-series ← new
# 15   debate  [CONTROL noop+chunks]   fresh noop with chunk time-series ← new
# ───────────────────────────────────────────────────────────────────────────
#
# Analyse results after completion:
#   for TRAIT in evil trustworthiness charismatic sycophantic loquacious ethical; do
#     python scripts/textarena/analyze_persona_probes.py \
#       --steered logs/mafia/mafia_activation_wolf_${TRAIT}_8p \
#       --control logs/mafia/mafia_noop_probe_8p \
#       --out reports/mafia_${TRAIT}_probe.txt
#   done
#
#   for TRAIT in charismatic sycophantic epistemic-humility sarcastic goal-oriented ethical; do
#     python scripts/textarena/analyze_persona_probes.py \
#       --steered logs/debate/debate_activation_p1_${TRAIT}_2p \
#       --control logs/debate/debate_noop_probe_2p \
#       --out reports/debate_${TRAIT}_probe.txt
#   done
#
# Submit:
#   sbatch scripts/textarena/slurm/run_persona_sweep_overnight.slurm

set -euo pipefail

NUM_SHARDS=5
RESUME_HOP_LIMIT=10
HOP=${1:-0}

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))
SHARD=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))

CONFIGS=(
    # ── Mafia wolf steering ─────────────────────────────────────────────────
    "config/games/mafia/mafia_activation_wolf_evil_8p.yaml"              #  0
    "config/games/mafia/mafia_activation_wolf_trustworthiness_8p.yaml"   #  1
    "config/games/mafia/mafia_activation_wolf_charismatic_8p.yaml"       #  2
    "config/games/mafia/mafia_activation_wolf_sycophantic_8p.yaml"       #  3
    "config/games/mafia/mafia_activation_wolf_loquacious_8p.yaml"        #  4
    "config/games/mafia/mafia_activation_wolf_ethical_8p.yaml"           #  5
    # ── Debate AGAINST steering ─────────────────────────────────────────────
    "config/games/debate/debate_activation_p1_charismatic_2p.yaml"       #  6
    "config/games/debate/debate_activation_p1_sycophantic_2p.yaml"       #  7
    "config/games/debate/debate_activation_p1_epistemic-humility_2p.yaml" # 8
    "config/games/debate/debate_activation_p1_sarcastic_2p.yaml"         #  9
    "config/games/debate/debate_activation_p1_goal-oriented_2p.yaml"     # 10
    "config/games/debate/debate_activation_p1_ethical_2p.yaml"           # 11
    # ── Controls (noop + probe, old format) ────────────────────────────────
    "config/games/mafia/mafia_noop_probe_8p.yaml"                        # 12
    "config/games/debate/debate_noop_probe_2p.yaml"                      # 13
    # ── Controls (noop + probe, new format with chunks) ─────────────────────
    "config/games/mafia/mafia_noop_chunks_8p.yaml"                       # 14
    "config/games/debate/debate_noop_chunks_2p.yaml"                     # 15
)

CONFIG="${CONFIGS[$CONFIG_IDX]}"
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} → config ${CONFIG_IDX} shard ${SHARD} (hop ${HOP})"
echo "[$(date)] Config: ${CONFIG}"

# ── Environment ────────────────────────────────────────────────────────────
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"

export PERSONA_VECTORS_ROOT="/scratch/inf0/user/nzukowsk/PEVectors/PEVectors/data/vector_extraction/persona/qwen3-14b"

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

# ── Run ────────────────────────────────────────────────────────────────────
"$PYTHON" scripts/run_episode.py \
    --config    "${CONFIG}" \
    --shard     "${SHARD}" \
    --num-shards "${NUM_SHARDS}"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} finished this chunk."

# ── Self-requeue if episodes remain ───────────────────────────────────────
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
    echo "[$(date)] ${REMAINING} episode(s) remaining — requeuing (hop $((HOP + 1)))"
    sbatch --dependency=afterany:"${SLURM_JOB_ID}" \
           --array="${SLURM_ARRAY_TASK_ID}" \
           "$0" $((HOP + 1))
elif [ "${REMAINING}" -gt 0 ]; then
    echo "[$(date)] WARNING: hop limit reached with ${REMAINING} remaining."
else
    echo "[$(date)] All episodes complete for config ${CONFIG_IDX} shard ${SHARD}."
fi
