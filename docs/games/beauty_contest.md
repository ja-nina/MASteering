# Keynesian Beauty Contest

**Family:** `symbolic` | **ID:** `beauty_contest`

---

## Rules

N players each guess an integer in `[low, high]` (default: `[0, 100]`). After all
guesses are submitted, the target is computed as:

```
target = floor(2/3 × mean(all guesses))
```

The player(s) closest to the target share the reward equally. The episode runs for
`num_rounds` rounds with no early stopping.

**Reward:** 1 / n_tied (split equally among tied winners), 0 for all others.

---

## Action format

> Respond with `CHOICE: <integer>` where `<integer>` ∈ [0, 100].

The parser also accepts a bare integer as a fallback.

---

## Config keys (`env_kwargs`)

| Key | Default | Description |
|---|---|---|
| `num_rounds` | 10 | Number of rounds to play |
| `low` | 0 | Minimum valid guess |
| `high` | 100 | Maximum valid guess |

---

## Experiments

### Reasoning sweep (`config/experiments/reasoning_sweep/`)

Tests whether reasoning effort (thinking tokens, step-by-step prompting) and ToM
instructions affect coordination.  Beauty contest is one of two games in this sweep;
the other is GBS.

```bash
python scripts/experiments/reasoning_sweep/gen_reasoning_sweep_configs.py
sbatch scripts/gbs/slurm/run_reasoning_sweep.slurm
python scripts/experiments/reasoning_sweep/plot_reasoning_sweep.py
```

Conditions include: `noop`, `non_thinking`, `thinking`, `noop_thinking`, `tom`,
`tom_thinking` across Qwen3-4B and gpt-oss-20b.

### Activation steering (individual configs)

Direct CAA vector extraction:

```bash
python scripts/gbs/steering/extract_steering_vector.py \
    --model Qwen/Qwen3-4B --layer model.layers.18 \
    --game beauty_contest --num-samples 64 \
    --output vectors/tom_beauty_contest_l18.npy
```

Individual hand-written steering configs are in `config/games/beauty_contest/`.

### Layer sweep

```bash
sbatch --array=10-35%8 scripts/gbs/slurm/run_layer_sweep.slurm beauty_contest
```

---

## Implementation files

| Purpose | File |
|---|---|
| Game logic | `testbed/envs/symbolic/beauty_contest.py` |
| Prompt renderer | `testbed/renderers/symbolic/beauty_contest.py` |
| Action parser | `testbed/parsers/symbolic/beauty_contest.py` |
| Registry entry | `testbed/registry.py` |

---

## Notes

- The target is re-computed each round from that round's mean, not from an initial
  target — so there is no fixed "correct" answer across rounds; convergence is a
  moving-target coordination problem.
- Unlike GBS there is no early termination on success; the episode always plays out
  all `num_rounds` rounds.
- No persona system is implemented for beauty contest yet.  See `gbs.md` for how
  personas work, and `adding_a_new_game.md` for how to add persona support.
