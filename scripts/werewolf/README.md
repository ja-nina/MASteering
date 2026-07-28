# Werewolf scripts

Placeholder for Werewolf-specific experiment scripts.

Suggested structure:

```
scripts/werewolf/
  gen_werewolf_configs.py        # sweep over player counts, roles, discussion lengths
  plot_werewolf_results.py       # win rate by role, vote accuracy, elimination patterns
  extract_werewolf_cases.py      # extract interesting day-vote situations for analysis
  slurm/
    run_werewolf_sweep.slurm
```

See `docs/games/werewolf.md` and `docs/games/adding_a_new_game.md`.
