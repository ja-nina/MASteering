# Adding a New Game

Follow these steps to wire a new game `<game_id>` into the testbed. The entire framework is game-agnostic; adding a game requires no changes to the orchestrator, policy, logging, or steering code.

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
```

`StepResult(rewards, done, info)` — set `done=True` on terminal steps.  `info` is
logged verbatim to JSONL and can carry anything diagnostic.

### `TextRenderer` contract (`testbed/renderers/base.py`)

```python
class MyGameRenderer:
    def system_prompt(self, agent_id: str, raw_obs: Any) -> str: ...
    def render(self, raw_obs: Any, agent_id: str, context: RenderContext) -> str: ...
```

`context.history` is a list of per-step dicts appended by the orchestrator;
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

## 3 — Add configs under `config/games/<game_id>/`

Create at minimum one YAML:

```yaml
# config/games/my_game/exp_noop.yaml
run_id: my_game_noop_2p
game:
  family: symbolic
  id: my_game
  env_kwargs:
    num_rounds: 10
episodes: 50
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
  dir: logs/my_game/
```

For sweep configs (many conditions), write a generator script in
`scripts/my_game/gen_<sweep_name>_configs.py` that writes to
`config/experiments/<sweep_name>/`.

---

## 4 — Create data directories

```
cases/my_game/           # extracted game states for counterfactual studies
logs/my_game/            # auto-created at runtime by run_episode.py
plots/my_game/           # analysis outputs
```

---

## 5 — Add game-specific scripts under `scripts/my_game/`

```
scripts/my_game/
  gen_<sweep>_configs.py    # config generator
  plot_<sweep>.py           # result visualisation
  slurm/                    # SLURM array jobs if running on a cluster
```

Keep `scripts/run_episode.py` and `scripts/analyze_results.py` unchanged — they are
game-agnostic entry points.

---

## 6 — Write tests

```
tests/envs/symbolic/test_my_game.py      # adapter unit tests
tests/parsers/test_my_game_parser.py     # parser unit tests
```

Use `StubPolicy` (`testbed/policy/base.py`) to test multi-round episodes without
loading a real model.

---

## 7 — Document the game

Add `docs/games/<game_id>.md` following the template of `gbs.md` or
`beauty_contest.md`.  Include: rules, action format, reward signal, available
config keys, and which experiments have been run.

---

## Quick smoke test

```bash
python scripts/run_episode.py \
  --config config/games/my_game/exp_noop.yaml \
  --episodes 1
```

The episode log lands in `logs/my_game/<run_id>/episode_0.jsonl`.
