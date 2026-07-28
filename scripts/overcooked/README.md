# Overcooked scripts

Placeholder for Overcooked-specific experiment scripts.

Suggested structure:

```
scripts/overcooked/
  gen_overcooked_configs.py      # sweep over layouts, personas, ToM conditions
  plot_overcooked_results.py     # score distributions, action type breakdown, coordination metrics
  slurm/
    run_overcooked_sweep.slurm
```

See `docs/games/overcooked.md` and `docs/games/adding_a_new_game.md`.
