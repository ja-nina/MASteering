# Goldstone Group Sum (GBS)

**Family:** `symbolic` | **ID:** `gbs`

## Rules

N players simultaneously submit a non-negative integer. The contributions are summed and
compared to a hidden target T. After each round every player learns the group sum and
the feedback below, but never the other players' contributions. The episode ends when the
group sum equals T, or after `num_rounds` rounds.

**Target:** `T ~ Uniform(low, high)`, seeded per episode (default `low = 5N`, `high = 50N`).
**Reward:** 1.0 for everyone on the round the group hits T, 0.0 otherwise.

Reference: Goldstone et al. (2024), *The emergence of specialized roles within groups*.

## Variants

| `env_kwargs` | Agents see | Action |
|---|---|---|
| default (`feedback: exact`) | group sum, signed error (`too HIGH by 23`) and N | `NUMBER: <int>` |
| `feedback: directional` | group sum and direction only | `NUMBER: <int>` |
| `hide_group_size: true` | Picking variant of Riedl (2025, arXiv 2510.05174), plain condition: no N, own guesses + direction only | `FINAL GUESS: <0-50>` |

## Config keys (`env_kwargs`)

| Key | Default | Description |
|---|---|---|
| `num_rounds` | 10 | Maximum rounds |
| `low`, `high` | `5N`, `50N` | Target range |
| `target` | random | Fixed target |
| `feedback` | `exact` | `exact` or `directional` |
| `hide_group_size` | `false` | Picking variant |

The main sweep (`config/sweeps/main.yaml`) runs `gbs` at 3 and 6 players and the picking
variant (`gbs_picking`) at 5. Steering the group (`select: all`), one member (`select: one`) or
half of it (`select: fraction`) asks whether a trait changes how fast the group specialises
and converges.
