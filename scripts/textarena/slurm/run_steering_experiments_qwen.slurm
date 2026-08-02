#!/bin/bash
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --array=0-39%10
#SBATCH -o /scratch/inf0/user/nzukowsk/MASteer/logs/steering_%A_%a.out
#SBATCH -e /scratch/inf0/user/nzukowsk/MASteer/logs/steering_%A_%a.err
#SBATCH -t 10:00:00
#
# Persona vector steering experiments + probed controls.
#
# Encoding: task_id = config_index * NUM_SHARDS + shard
# 4 configs × 10 shards = 40 tasks; 10 concurrent max (%10).
#
# Config index → experiment:
#   0  mafia  steered (wolf → opportunistic, adaptive)
#   1  mafia  control (noop + probe)
#   2  debate steered (player_1 AGAINST → assertive, adaptive)
#   3  debate control (noop + probe)
#
# Submit:
#   sbatch scripts/textarena/slurm/run_steering_experiments_qwen.slurm
#
# Analyse results after completion:
#   python scripts/textarena/analyze_persona_probes.py \
#       --steered logs/mafia/mafia_activation_wolf_opportunistic_8p \
#       --control logs/mafia/mafia_noop_probe_8p \
#       --out reports/mafia_probe_analysis.txt
#
#   python scripts/textarena/analyze_persona_probes.py \
#       --steered logs/debate/debate_activation_p1_assertive_2p \
#       --control logs/debate/debate_noop_probe_2p \
#       --out reports/debate_probe_analysis.txt

set -euo pipefail

NUM_SHARDS=10
RESUME_HOP_LIMIT=20
HOP=${1:-0}

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SHARDS ))
SHARD=$(( SLURM_ARRAY_TASK_ID % NUM_SHARDS ))

CONFIGS=(
    "config/games/mafia/mafia_activation_wolf_opportunistic_8p.yaml"   # 0
    "config/games/mafia/mafia_noop_probe_8p.yaml"                      # 1
    "config/games/debate/debate_activation_p1_assertive_2p.yaml"       # 2
    "config/games/debate/debate_noop_probe_2p.yaml"                    # 3
)

CONFIG="${CONFIGS[$CONFIG_IDX]}"
echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} → config ${CONFIG_IDX} shard ${SHARD} (hop ${HOP})"
echo "[$(date)] Config: ${CONFIG}"

# ── Environment ────────────────────────────────────────────────────────────────
PYTHON="/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python"

# HF model cache — must be OUTSIDE the repo to avoid Lustre Errno 70
export HF_HOME="/scratch/inf0/user/nzukowsk/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:-}"   # set in your ~/.bashrc or pass via sbatch --export

# Persona vectors — point at the PersVecGen extraction outputs
export PERSONA_VECTORS_ROOT="/scratch/inf0/user/nzukowsk/PEVectors/PEVectors/data/vector_extraction/persona/qwen3-14b"

mkdir -p /scratch/inf0/user/nzukowsk/MASteer/logs

# ── Run ────────────────────────────────────────────────────────────────────────
"$PYTHON" scripts/run_episode.py \
    --config "${CONFIG}" \
    --shard  "${SHARD}" \
    --num-shards "${NUM_SHARDS}"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} finished this chunk."

# ── Self-requeue if episodes remain ───────────────────────────────────────────
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
