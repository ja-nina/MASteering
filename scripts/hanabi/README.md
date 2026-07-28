# Hanabi scripts

Placeholder for Hanabi-specific experiment scripts.

Suggested structure:

```
scripts/hanabi/
  gen_hanabi_configs.py          # sweep over player counts, hint strategies, personas
  plot_hanabi_results.py         # score distributions, hint efficiency, fuse rate
  slurm/
    run_hanabi_sweep.slurm
```

See `docs/games/hanabi.md` and `docs/games/adding_a_new_game.md`.
