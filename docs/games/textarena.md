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
| `Diplomacy-v0` | Multi-player negotiation |
| `Negotiation-v0` | Bilateral deal-making |
| `SecretMafia-v0` | Hidden-role elimination |
| `Taboo-v0` | Constrained word-guessing |
| `ThreePlayerGOPS-v0` | Card-game strategy |
| `ThreePlayerIPD-v0` | Iterated Prisoner's Dilemma |
| `ThreePlayerTicTacToe-v0` | Tic-tac-toe variant |
| `TwoRoomsAndABoom-v0` | Hidden-role coordination |

---

## Config

```yaml
run_id: textarena_diplomacy_2p
game:
  family: textarena
  id: Diplomacy-v0    # any valid TextArena env_id
  env_kwargs: {}
episodes: 20
model:
  backend: transformers
  model_id: Qwen/Qwen3-14B
  enable_thinking: false
  temperature: 0.7
  top_p: 0.8
  top_k: 20
  min_p: 0.0
agents:
  count: 2
  concurrency: sequential
  max_parse_retries: 5
steering:
  default: noop
  per_agent: {}
logging:
  dir: logs/textarena/
```

Place configs in `config/games/textarena/`.

---

## How the adapter works

`TextArenaAdapter` wraps the `textarena` gym API:

- `reset()` → calls `ta_env.reset(num_players=N)`
- `pending()` → calls `ta_env.get_observation()` to get the current agent's turn
- `submit({agent_id: action})` → calls `ta_env.step(action)`

TextArena produces observation strings natively; the renderer passes them through
unchanged and the parser accepts any non-empty string.

---

## Adding a new TextArena game

No code changes are required. Just add a config file:

```bash
cp config/games/textarena/example_noop.yaml \
   config/games/textarena/my_game_noop.yaml
# Edit: set id: MyGame-v0
python scripts/run_episode.py --config config/games/textarena/my_game_noop.yaml --episodes 1
```

---

## Implementation files

| Purpose | File |
|---|---|
| Adapter | `testbed/envs/textarena/ta_adapter.py` |
| Renderer | `testbed/renderers/textarena.py` |
| Parser | `testbed/parsers/textarena.py` |
| Registry entry | `testbed/registry.py` (`family == "textarena"` branch) |
