# Keynesian beauty contest

**Family:** `symbolic` | **ID:** `beauty_contest`

## Rules

Every round each player simultaneously picks an integer in `[low, high]`. The target is
2/3 of the average pick; the player closest to it wins the round (ties share the reward).
After each round players see the average, the target and whether their own guess was too
high, too low or a win. The game is deterministic, so `seed` is accepted but unused.

**Action:** `CHOICE: <integer>` (the keyword is required).

## Config keys (`env_kwargs`)

| Key | Default | Description |
|---|---|---|
| `num_rounds` | 5 | Rounds per episode |
| `low`, `high` | 0, 100 | Range of valid picks |

## Research angle

The level-k depth of reasoning shows in how fast picks fall toward 0. Steering one player
(`select: one`) towards e.g. `assertive` or `curious` tests whether a trait changes its depth of
reasoning, and whether the unsteered players adapt to it.
