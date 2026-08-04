#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --array=0-99%20
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/rotation_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/rotation_%A_%a.err
#SBATCH -t 04:00:00
#
# ════════════════════════════════════════════════════════════════════════════
#  ORBIT-style rotation steering sweep — 18 traits × 2 games + 2 noop controls
#  mode: rotation, coefficient: 0.3 rad (≈ 17°, norm-preserving)
#  20 configs × 5 shards = 100 tasks  |  %20 concurrent
#
#  A/B partner to run_additive_sweep.slurm — identical configs except mode.
#  Reference: Aparin & Gaintseva (arXiv 2606.06735); ORBIT (arXiv 2606.22357)
# ════════════════════════════════════════════════════════════════════════════
#
# Configs
# ───────────────────────────────────────────────────────────────────────────
# idx  game    trait
#  0   debate  assertive
#  1   debate  charismatic
#  2   debate  epistemic-humility
#  3   debate  ethical
#  4   debate  goal-oriented
#  5   debate  sarcastic
#  6   debate  sycophantic
#  7   debate  apathetic
#  8   debate  angry
#  9   mafia   charismatic
# 10   mafia   ethical
# 11   mafia   evil
# 12   mafia   loquacious
# 13   mafia   opportunistic
# 14   mafia   sycophantic
# 15   mafia   trustworthiness
# 16   mafia   apathetic
# 17   mafia   angry
# 18   debate  [CONTROL noop+chunks — probes layers 10,20,29]
# 19   mafia   [CONTROL noop+chunks — probes layers 10,20,29]
# ───────────────────────────────────────────────────────────────────────────
#
# Submit:
#   sbatch scripts/textarena/slurm/run_rotation_sweep.slurm
#
# Compare with additive results:
#   python scripts/textarena/outcome_analysis.py \
#       --logs logs/debate logs/mafia \
#       --mode rotation additive noop

set -euo pipefail

NUM_SHARDS=5
RESUME_HOP_LIMIT=10
HOP=${1:-0}

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))
SHARD=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))

CONFIGS=(
    # ── Debate rotation steering ─────────────────────────────────────────────
    "config/games/debate/debate_activation_p1_assertive_rotation_2p.yaml"          #  0
    "config/games/debate/debate_activation_p1_charismatic_rotation_2p.yaml"        #  1
    "config/games/debate/debate_activation_p1_epistemic-humility_rotation_2p.yaml" #  2
    "config/games/debate/debate_activation_p1_ethical_rotation_2p.yaml"            #  3
    "config/games/debate/debate_activation_p1_goal-oriented_rotation_2p.yaml"      #  4
    "config/games/debate/debate_activation_p1_sarcastic_rotation_2p.yaml"          #  5
    "config/games/debate/debate_activation_p1_sycophantic_rotation_2p.yaml"        #  6
    "config/games/debate/debate_activation_p1_apathetic_rotation_2p.yaml"          #  7
    "config/games/debate/debate_activation_p1_angry_rotation_2p.yaml"              #  8
    # ── Mafia rotation steering ──────────────────────────────────────────────
    "config/games/mafia/mafia_activation_wolf_charismatic_rotation_8p.yaml"        #  9
    "config/games/mafia/mafia_activation_wolf_ethical_rotation_8p.yaml"            # 10
    "config/games/mafia/mafia_activation_wolf_evil_rotation_8p.yaml"               # 11
    "config/games/mafia/mafia_activation_wolf_loquacious_rotation_8p.yaml"         # 12
    "config/games/mafia/mafia_activation_wolf_opportunistic_rotation_8p.yaml"      # 13
    "config/games/mafia/mafia_activation_wolf_sycophantic_rotation_8p.yaml"        # 14
    "config/games/mafia/mafia_activation_wolf_trustworthiness_rotation_8p.yaml"    # 15
    "config/games/mafia/mafia_activation_wolf_apathetic_rotation_8p.yaml"          # 16
    "config/games/mafia/mafia_activation_wolf_angry_rotation_8p.yaml"              # 17
    # ── Noop baselines (shared with additive sweep) ──────────────────────────
    "config/games/debate/debate_noop_chunks_2p.yaml"                               # 18
    "config/games/mafia/mafia_noop_chunks_8p.yaml"                                 # 19
)

CONFIG="${CONFIGS[$CONFIG_IDX]}"
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} → config ${CONFIG_IDX} shard ${SHARD} (hop ${HOP})"
echo "[$(date)] Config: ${CONFIG}"

# ── Environment ────────────────────────────────────────────────────────────
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"
export PERSONA_VECTORS_ROOT="/scratch/inf0/user/nzukowsk/PEVectors/PEVectors/data/vector_extraction/persona/qwen3-14b"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set before sbatch submission}"

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

# ── Run ────────────────────────────────────────────────────────────────────
"$PYTHON" scripts/run_episode.py \
    --config    "${CONFIG}" \
    --shard     "${SHARD}" \
    --num-shards "${NUM_SHARDS}"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} finished."

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
