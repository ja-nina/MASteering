# Adding a New Game

Follow these steps to wire a new game `<game_id>` into the testbed. The entire framework is game-agnostic; adding a game requires no changes to the runner, engines, logging, or steering code.

---

## 1 — Implement the triad: Adapter + Renderer + Parser

Create three files under the appropriate family directory.

**For a new symbolic (simultaneous-move) game:**

```
testbed/envs/symbolic/<game_id>.py      # game logic
testbed/renderers/symbolic/<game_id>.py  # state → prompt text
testbed/parsers/symbolic/<game_id>.py    # text → action
```

**For a TextArena game** — no new files are needed. The `TextArenaAdapter`, `TextArenaRenderer`, and `TextArenaParser` handle all TextArena game IDs automatically via the `env_id` config key.

### `EnvAdapter` contract (`testbed/envs/adapter.py`)

```python
class MyGameAdapter(SymbolicAdapter):
    def reset(self, seed=None) -> None: ...
    def pending(self) -> list[tuple[str, Any]]: ...   # which agents act this step
    def submit(self, actions: dict[str, Any]) -> StepResult: ...
    def agent_ids(self) -> list[str]: ...
    def legal_actions(self, agent_id: str) -> Any: ...
    def close(self) -> dict: ...   # return per-agent final rewards
    # hidden-role games only (enables `team:` / `roles:` steering groups):
    def roles(self) -> dict[str, str]: ...   # {agent_id: role}
    def teams(self) -> dict[str, str]: ...   # {agent_id: "good" | "evil"}
```

The constructor must accept `seed` (the runner passes a per-episode seed that depends only on
game, player count and episode index, so every steering condition plays the same games).
`pending()` may return one agent (turn-based) or several (simultaneous); the runner batches
whatever is pending across all concurrent episodes.

`StepResult(rewards, done, info)` — set `done=True` on terminal steps.  `info` is
logged verbatim to JSONL and can carry anything diagnostic.

### `TextRenderer` contract (`testbed/renderers/base.py`)

```python
class MyGameRenderer:
    def system_prompt(self, agent_id: str, raw_obs: Any) -> str: ...
    def render(self, raw_obs: Any, agent_id: str, context: RenderContext) -> str: ...
```

`context.history` is a list of per-step dicts appended by the adapter;
use it to show round-history summaries.

### `ActionParser` contract (`testbed/parsers/base.py`)

```python
class MyGameParser:
    def parse(self, completion: str, raw_obs: Any,
              agent_id: str, context: RenderContext) -> ParseResult:
        # return ParsedAction(value) on success
        # return ParseError(feedback_string) on failure — triggers a re-prompt
```

---

### Optional: an image view

To support `observation: image | both` (see [observations.md](observations.md)), the renderer adds:

```python
class MyGameRenderer:
    modes = ("text", "image", "both")
    def render_images(self, raw_obs, agent_id, context) -> list:   # PIL images of the state
    def render_visual(self, raw_obs, agent_id, context) -> str:    # render() minus what the images show
```

`testbed/visual.py` has helpers for labelled grids, bar charts, rings of players and cards.

---

## 2 — Register the game in `testbed/registry.py`

Add your triad to the `_SYMBOLIC` dict:

```python
from testbed.envs.symbolic.my_game import MyGameAdapter
from testbed.renderers.symbolic.my_game import MyGameRenderer
from testbed.parsers.symbolic.my_game import MyGameParser

_SYMBOLIC = {
    ...
    "my_game": (MyGameAdapter, MyGameRenderer, MyGameParser),
}
```

---

## 3 — Add it to a sweep

Add the game to `config/sweeps/main.yaml` (or a new sweep file) under `games:`:

```yaml
games:
  my_game: {family: symbolic, id: my_game, players: [4], env_kwargs: {num_rounds: 8}}
  # roles: true  -> also generate the team/role steering conditions
```

Then `python scripts/make_configs.py config/sweeps/main.yaml` writes one run config per
condition × trait × coefficient to `config/runs/main/my_game_4p/`.

---

## 4 — Test without a GPU

`tests/fakes.py` has a `FakeGenerator` (scripted replies) and a fake PersVecFast vector
directory; `tests/test_runner.py` shows how to play full episodes with them. Add the game's
keyword to `fakes.reply` and write rules tests under `tests/envs/symbolic/`.

---

## 5 — Document

Add `docs/games/<game_id>.md`: rules, action format, `env_kwargs`, research angle.
