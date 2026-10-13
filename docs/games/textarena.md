# TextArena Games

**Family:** `textarena` | **ID:** any valid TextArena `env_id`

---

## Overview

TextArena is a turn-based multi-agent game library with a gym-style API.
The testbed wraps it via `TextArenaAdapter`, `TextArenaRenderer`, and
`TextArenaParser` — no game-specific adapter code is needed.

---

## Supported games

Any `env_id` accepted by `textarena.make()` works out of the box. Known examples
from the TextArena library:

| env_id | Description |
|---|---|
| `BlindAuction-v0` | Sealed-bid auction |
| `CharacterConclave-v0` | Role-deduction coordination |
| `Codenames-v0` | Word-clue deduction |
| `Negotiation-v0` | Bilateral deal-making |
| `SecretMafia-v0` | Hidden-role elimination |
| `Taboo-v0` | Constrained word-guessing |
| `ThreePlayerGOPS-v0` | Card-game strategy |
| `ThreePlayerIPD-v0` | Iterated Prisoner's Dilemma |
| `ThreePlayerTicTacToe-v0` | Tic-tac-toe variant |
| `TwoRoomsAndABoom-v0` | Hidden-role coordination |

---

## Config

TextArena games are listed in `config/sweeps/main.yaml` like any other game:

```yaml
games:
  mafia:  {family: textarena, id: SecretMafia-v0, players: [8], roles: true}
  debate: {family: textarena, id: Debate-v0, players: [2],
           env_kwargs: {jury_model: meta-llama/llama-3.3-70b-instruct, jury_size: 1}}
```

`jury_model` / `jury_size` replace TextArena's jury with one fixed OpenRouter model
(needs `OPENROUTER_API_KEY` in `.env`). `roles: true` enables team steering: `roles()` reads
TextArena's per-seat roles, and `teams()` marks the deceiving side (mafia, werewolves, spies, …)
as `evil`.

---

## How the adapter works

`TextArenaAdapter` wraps the `textarena` gym API:

- `reset()` → calls `ta_env.reset(num_players=N, seed=episode_seed)`
- `pending()` → calls `ta_env.get_observation()` to get the current agent's turn
- `submit({agent_id: action})` → calls `ta_env.step(action)`

TextArena produces observation strings natively; the renderer passes them through
unchanged and the parser accepts any non-empty string.

---

## Adding a new TextArena game

No code changes are required: add the env id to a sweep file under `games:` and regenerate
the configs with `python scripts/make_configs.py <sweep>`.

---

## Implementation files

| Purpose | File |
|---|---|
| Adapter | `testbed/envs/textarena/ta_adapter.py` |
| Renderer | `testbed/renderers/textarena.py` |
| Parser | `testbed/parsers/textarena.py` |
| Registry entry | `testbed/registry.py` (`family == "textarena"` branch) |
