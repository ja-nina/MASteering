# Werewolf

**Family:** `symbolic` | **ID:** `werewolf` | **Status:** skeleton — `submit()` not yet implemented

---

## Game overview

Werewolf (also known as Mafia) is a social deduction game with **hidden roles** and
alternating day/night phases.  A small group of Werewolves hides among a larger
group of Villagers.  During the night Werewolves secretly choose a player to
eliminate; during the day all surviving players discuss, argue, and vote to
eliminate one suspect.

**Villagers win** if all Werewolves are eliminated.
**Werewolves win** if they reach parity with (or outnumber) the village.

Special village roles — Seer (investigates one player per night), Doctor
(saves one player per night) — add information asymmetry that the LLM agents
must reason about and conceal or exploit.

---

## Phase structure

Each day/night cycle runs through these phases:

| Phase | `pending()` returns | Action format |
|---|---|---|
| `night_werewolf` | all living werewolves | `KILL: player_X` |
| `night_seer` | seer (if alive) | `INVESTIGATE: player_X` |
| `night_doctor` | doctor (if alive) | `SAVE: player_X` (can save self) |
| `day_discussion` | one speaker per step (round-robin) | `STATEMENT: <free text>` |
| `day_vote` | all living | `VOTE: player_X` |

Night phases where the role has been eliminated are skipped automatically.
Discussion repeats for `discussion_rounds_per_day` complete rotations before
advancing to the vote.

---

## Role roster (default by player count)

| Players | Werewolves | Village roles |
|---|---|---|
| 4 | 1 | Seer + 2 Villagers |
| 5 | 1 | Seer + 3 Villagers |
| 6 | 2 | Seer + Doctor + 2 Villagers |
| 7–8 | 2 | Seer + Doctor + 3–4 Villagers |
| 9–10 | 3 | Seer + Doctor + 4–5 Villagers |

Pass `roles=["Werewolf", "Seer", "Villager", ...]` to override.

---

## Config keys (`env_kwargs`)

| Key | Default | Description |
|---|---|---|
| `num_players` | 6 | 4–10 players |
| `discussion_rounds_per_day` | 1 | Complete speaking rotations before day vote |
| `roles` | `None` | Override role list; length must equal `num_players` |
| `seed` | 0 | RNG seed for role assignment |

---

## Implementation status

| File | Status |
|---|---|
| `testbed/envs/symbolic/werewolf.py` | State + phase structure done; 5 phase branches in `submit()` need implementing |
| `testbed/renderers/symbolic/werewolf.py` | Complete — role-aware system prompt + per-phase render |
| `testbed/parsers/symbolic/werewolf.py` | Complete — parses KILL/INVESTIGATE/SAVE/STATEMENT/VOTE |
| `testbed/registry.py` | Lazy-loaded via `_SYMBOLIC_LAZY["werewolf"]` |

**To activate:** implement the 5 `raise NotImplementedError` branches in `submit()`, then un-comment `"werewolf"` in `testbed/registry.py`.

---

## Note on TextArena alternative

TextArena's `SecretMafia-v0` covers similar social deduction mechanics and is
fully playable today via `family: textarena`.  The custom adapter here gives
finer control: explicit Seer/Doctor roles, configurable discussion length, and
structured vote recording for analysis.

---

## Key research questions

- Do Werewolf agents successfully deceive in day discussions, or do they reveal behavioural tells?
- Can a ToM instruction help a Seer reason about whether their investigation target is being saved vs. targeted?
- Do certain personas (e.g., `contrarian`, `chaotic`) correlate with vote outcomes independent of role?
- How does group size affect the Werewolves' win rate with LLM agents?

---

## Quick smoke test (after implementing submit)

```bash
python scripts/run_episode.py \
  --config config/games/werewolf/werewolf_noop_6p.yaml \
  --episodes 1
```
