# Diplomacy scripts

Placeholder for Diplomacy-specific experiment scripts.

Diplomacy is playable today via the TextArena `Diplomacy-v0` adapter with no additional code.

Suggested structure for analysis scripts:

```
scripts/diplomacy/
  gen_diplomacy_configs.py       # sweep over player counts, personas, ToM conditions
  plot_diplomacy_results.py      # supply centre counts, alliance patterns, survival rates
  slurm/
    run_diplomacy_sweep.slurm
```

See `docs/games/diplomacy.md` and `docs/games/adding_a_new_game.md`.
