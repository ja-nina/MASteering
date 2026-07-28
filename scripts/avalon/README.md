# Avalon scripts

Placeholder for Avalon-specific experiment scripts.

Suggested structure once experiments are designed:

```
scripts/avalon/
  gen_avalon_configs.py          # sweep over roles, player counts, personas
  plot_avalon_results.py         # win rate, deception success, vote patterns
  extract_avalon_cases.py        # extract interesting game states for counterfactual study
  slurm/
    run_avalon_sweep.slurm
```

See `docs/games/avalon.md` and `docs/games/adding_a_new_game.md`.
