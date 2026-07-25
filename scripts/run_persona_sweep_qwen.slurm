#!/bin/bash
#SBATCH --job-name=persona_sweep_qwen
#SBATCH --output=logs/slurm/persona_sweep_%A_%a.out
#SBATCH --error=logs/slurm/persona_sweep_%A_%a.err
#SBATCH --array=0-20          # 21 conditions (plain + 20 behavioural personas)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue

# ── Self-requeue on walltime (same pattern as picking sweep) ────────────────
RESUME_HOP_LIMIT=${RESUME_HOP_LIMIT:-20}
HOP=${HOP:-0}
if [ "$HOP" -ge "$RESUME_HOP_LIMIT" ]; then
    echo "Reached hop limit ($RESUME_HOP_LIMIT). Exiting."
    exit 0
fi

requeue_self() {
    echo "Walltime approaching — requeuing (hop $((HOP+1)))."
    sbatch --export=ALL,HOP=$((HOP+1)) "$0"
}
trap requeue_self USR1

# ── Map array index → config file ───────────────────────────────────────────
CONFIGS=(
    persona_impact_plain_2p_14b
    persona_impact_cooperative_2p_14b
    persona_impact_helpful_2p_14b
    persona_impact_sycophantic_2p_14b
    persona_impact_analytical_2p_14b
    persona_impact_game_theorist_2p_14b
    persona_impact_risk_averse_2p_14b
    persona_impact_risk_seeking_2p_14b
    persona_impact_competitive_2p_14b
    persona_impact_confident_2p_14b
    persona_impact_uncertain_2p_14b
    persona_impact_optimistic_2p_14b
    persona_impact_pessimistic_2p_14b
    persona_impact_contrarian_2p_14b
    persona_impact_leader_2p_14b
    persona_impact_mediator_2p_14b
    persona_impact_scientist_2p_14b
    persona_impact_intuitive_2p_14b
    persona_impact_social_2p_14b
    persona_impact_overconfident_2p_14b
    persona_impact_chaotic_2p_14b
)

RUN_ID=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CONFIG="config/persona_sweep/${RUN_ID}.yaml"

echo "Array task $SLURM_ARRAY_TASK_ID → $RUN_ID"
mkdir -p logs/slurm logs/persona_sweep

python scripts/run_episode.py --config "$CONFIG" &
wait $!
