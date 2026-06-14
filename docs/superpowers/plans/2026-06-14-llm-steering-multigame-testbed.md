# LLM Steering Multi-Game Testbed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testbed that runs small local LLM agents as multiple agents across text-based multi-agent games (symbolic + TextArena), with a fully-implemented steering layer (no-op, prompt-injection, activation vectors via forward hooks).

**Architecture:** A single orchestrator drives any game through a generalized `EnvAdapter` (`pending()`/`submit()`, covering simultaneous and turn-based games). Each game has a `TextRenderer` (state→prompt) and `ActionParser` (text→action, with error feedback). A `Policy` (transformers in-process for steering, or vLLM client for fast baselines) produces completions, and a `SteeringMethod` modifies prompts and/or activations. A `registry` maps `game_id → (adapter, renderer, parser)`.

**Tech Stack:** Python 3.9, stdlib dataclasses, numpy, pyyaml, pytest. torch + transformers for the steering-capable policy (GPU at run time; hook math tested on CPU toy module). textarena for the TextArena adapter.

**Environment notes for the executor:**
- Run tests with the base conda python: `python -m pytest` (it has numpy, pyyaml, torch-cpu, pytest).
- This machine is **CPU-only with old transformers (4.20.1)** — tests that load Qwen or need a GPU are marked `@pytest.mark.gpu` and skipped here. The activation-steering *mechanism* is tested on a tiny CPU toy `nn.Module`, which DOES run here.
- `textarena` may not be installed; its real-integration test uses `pytest.importorskip("textarena")`. Our adapter is unit-tested against a fake env so logic is covered without the lib.
- Windows shell is PowerShell; commit commands below use git which works cross-platform.

---

## File Structure

```
testbed/
├── __init__.py
├── types.py                  # RawObs, Action, StepResult, ParseResult, ParsedAction, ParseError, SteeringSpec, RenderContext
├── envs/
│   ├── __init__.py
│   ├── adapter.py            # EnvAdapter Protocol
│   ├── symbolic/
│   │   ├── __init__.py
│   │   ├── base.py           # SymbolicAdapter base (simultaneous turn model)
│   │   ├── beauty_contest.py # BeautyContestAdapter
│   │   └── gbs.py            # GBSAdapter (group binary search)
│   └── textarena/
│       ├── __init__.py
│       └── ta_adapter.py     # TextArenaAdapter (turn-based)
├── renderers/
│   ├── __init__.py
│   ├── base.py               # TextRenderer Protocol
│   ├── symbolic/
│   │   ├── __init__.py
│   │   ├── beauty_contest.py
│   │   └── gbs.py
│   └── textarena.py          # pass-through renderer
├── parsers/
│   ├── __init__.py
│   ├── base.py               # ActionParser Protocol
│   ├── symbolic/
│   │   ├── __init__.py
│   │   ├── beauty_contest.py
│   │   └── gbs.py
│   └── textarena.py          # near pass-through parser
├── policy/
│   ├── __init__.py
│   ├── base.py               # Policy Protocol + StubPolicy
│   ├── transformers_policy.py# in-process HF + steering hooks (gpu-gated test)
│   └── vllm_policy.py        # OpenAI-compatible client (gpu/server-gated test)
├── steering/
│   ├── __init__.py
│   ├── base.py               # SteeringMethod Protocol
│   ├── noop.py
│   ├── prompt_injection.py
│   └── activation.py         # vector load + forward-hook factory (CPU toy test)
├── logging_/
│   ├── __init__.py
│   └── episode_logger.py
├── registry.py               # game_id -> builder
└── orchestrator.py           # single game loop
config/
└── run_config.yaml
scripts/
└── run_episode.py
tests/
├── conftest.py
└── ... (mirrors testbed/)
requirements.txt
pytest.ini
```

---

## Task 1: Project scaffolding, requirements, pytest config

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `testbed/__init__.py`, `tests/__init__.py`, `tests/conftest.py`
- Create empty package `__init__.py` files listed in File Structure.

- [ ] **Step 1: Create `requirements.txt`**

```
numpy>=1.24
pyyaml>=6.0
pytest>=7.0
# steering-capable policy (run time; GPU recommended)
torch>=2.1
transformers>=4.45
accelerate>=0.30
# fast baseline backend
openai>=1.0
# turn-based games
textarena>=0.6
```

- [ ] **Step 2: Create `pytest.ini` registering the `gpu` marker**

```ini
[pytest]
markers =
    gpu: tests that require a GPU and a modern transformers/Qwen install (skipped in CPU-only envs)
testpaths = tests
```

- [ ] **Step 3: Create package `__init__.py` files** (all empty) for:
`testbed/__init__.py`, `testbed/envs/__init__.py`, `testbed/envs/symbolic/__init__.py`, `testbed/envs/textarena/__init__.py`, `testbed/renderers/__init__.py`, `testbed/renderers/symbolic/__init__.py`, `testbed/parsers/__init__.py`, `testbed/parsers/symbolic/__init__.py`, `testbed/policy/__init__.py`, `testbed/steering/__init__.py`, `testbed/logging_/__init__.py`, `tests/__init__.py`.

(Note: package is `logging_` with trailing underscore to avoid shadowing stdlib `logging`.)

- [ ] **Step 4: Create `tests/conftest.py`** ensuring repo root on path

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 5: Verify pytest collects nothing yet without error**

Run: `python -m pytest -q`
Expected: "no tests ran" (exit code 5) — confirms config is valid.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini testbed tests
git commit -m "chore: project scaffolding, requirements, pytest config"
```

---

## Task 2: Core types

**Files:**
- Create: `testbed/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
from testbed.types import (
    StepResult, ParsedAction, ParseError, SteeringSpec, RenderContext,
)


def test_step_result_fields():
    sr = StepResult(rewards={"player_0": 1.0}, done=False, info={"k": "v"})
    assert sr.rewards["player_0"] == 1.0
    assert sr.done is False
    assert sr.info["k"] == "v"


def test_parse_result_variants():
    ok = ParsedAction(value=42)
    err = ParseError(feedback="bad output, try again")
    assert ok.value == 42
    assert err.feedback == "bad output, try again"


def test_steering_spec_defaults():
    spec = SteeringSpec(method="activation", layer="model.layers.14",
                        vector_path="v.pt", coefficient=8.0)
    assert spec.method == "activation"
    assert spec.coefficient == 8.0
    noop = SteeringSpec(method="noop")
    assert noop.layer is None and noop.vector_path is None and noop.coefficient == 0.0


def test_render_context_history():
    ctx = RenderContext(round_index=2, history=[{"target": 27.5}],
                        last_rewards={"player_0": 0.0}, extra={})
    assert ctx.round_index == 2
    assert ctx.history[-1]["target"] == 27.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_types.py -q`
Expected: FAIL (ModuleNotFoundError: testbed.types)

- [ ] **Step 3: Write `testbed/types.py`**

```python
"""Core shared types for the testbed."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Game-specific payloads kept loose on purpose: symbolic games use dicts,
# TextArena uses the observation string, etc.
RawObs = Any
Action = Any


@dataclass
class StepResult:
    rewards: Dict[str, float]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedAction:
    value: Action


@dataclass
class ParseError:
    feedback: str


ParseResult = Union[ParsedAction, ParseError]


@dataclass
class SteeringSpec:
    method: str  # "noop" | "prompt_injection" | "activation"
    layer: Optional[str] = None
    vector_path: Optional[str] = None
    coefficient: float = 0.0
    # free-form params for prompt injection (e.g. {"system_suffix": "..."})
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderContext:
    round_index: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_rewards: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_types.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add testbed/types.py tests/test_types.py
git commit -m "feat: core shared types"
```

---

## Task 3: EnvAdapter protocol

**Files:**
- Create: `testbed/envs/adapter.py`
- Test: `tests/envs/test_adapter_protocol.py` (create `tests/envs/__init__.py`)

- [ ] **Step 1: Write the failing test** (a minimal concrete adapter must satisfy the protocol)

```python
from testbed.envs.adapter import EnvAdapter
from testbed.types import StepResult


class _Toy:
    def reset(self): self._done = False
    def pending(self): return [("player_0", {"obs": 1})]
    def submit(self, actions): return StepResult(rewards={"player_0": 1.0}, done=True)
    def agent_ids(self): return ["player_0"]
    def legal_actions(self, agent_id): return None
    def close(self): return {"player_0": 1.0}


def test_toy_satisfies_protocol():
    a: EnvAdapter = _Toy()
    a.reset()
    pend = a.pending()
    assert pend[0][0] == "player_0"
    res = a.submit({"player_0": 5})
    assert res.done is True
    assert isinstance(a.close(), dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/envs/test_adapter_protocol.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Create `tests/envs/__init__.py`** (empty) and write `testbed/envs/adapter.py`**

```python
"""The single environment abstraction unifying simultaneous and turn-based games."""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from testbed.types import Action, RawObs, StepResult


@runtime_checkable
class EnvAdapter(Protocol):
    def reset(self) -> None: ...

    def pending(self) -> List[Tuple[str, RawObs]]:
        """Agents that must act now. ALL for simultaneous games, ONE for turn-based."""
        ...

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        """Apply the pending agents' actions and advance the environment."""
        ...

    def agent_ids(self) -> List[str]: ...

    def legal_actions(self, agent_id: str) -> Optional[object]: ...

    def close(self) -> Dict[str, float]:
        """Terminate and return final rewards / game info."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/envs/test_adapter_protocol.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add testbed/envs/adapter.py tests/envs/__init__.py tests/envs/test_adapter_protocol.py
git commit -m "feat: EnvAdapter protocol"
```

---

## Task 4: Beauty Contest adapter

Rules: N players each submit an integer in [0, 100] per round. Target = (2/3) × mean(choices). The player(s) closest to the target win the round (reward 1.0, split on ties); others 0.0. Runs `num_rounds` rounds. Simultaneous turn model. History records each round's mean and target.

**Files:**
- Create: `testbed/envs/symbolic/base.py`, `testbed/envs/symbolic/beauty_contest.py`
- Test: `tests/envs/symbolic/test_beauty_contest.py` (create `tests/envs/symbolic/__init__.py`)

- [ ] **Step 1: Write the failing test**

```python
from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter


def test_pending_returns_all_players_simultaneously():
    env = BeautyContestAdapter(num_players=3, num_rounds=2)
    env.reset()
    pend = env.pending()
    assert sorted(a for a, _ in pend) == ["player_0", "player_1", "player_2"]


def test_target_and_winner():
    env = BeautyContestAdapter(num_players=3, num_rounds=1)
    env.reset()
    # choices 0, 60, 90 -> mean 50 -> target 33.33; closest is 60 (player_1)
    res = env.submit({"player_0": 0, "player_1": 60, "player_2": 90})
    assert res.done is True
    assert res.rewards["player_1"] == 1.0
    assert res.rewards["player_0"] == 0.0
    assert res.rewards["player_2"] == 0.0
    assert round(res.info["target"], 2) == 33.33


def test_tie_splits_reward():
    env = BeautyContestAdapter(num_players=2, num_rounds=1)
    env.reset()
    # both pick 30 -> mean 30 -> target 20; equal distance -> split 0.5 each
    res = env.submit({"player_0": 30, "player_1": 30})
    assert res.rewards["player_0"] == 0.5
    assert res.rewards["player_1"] == 0.5


def test_runs_multiple_rounds_then_done():
    env = BeautyContestAdapter(num_players=2, num_rounds=2)
    env.reset()
    r1 = env.submit({"player_0": 10, "player_1": 20})
    assert r1.done is False
    assert len(env.context.history) == 1
    r2 = env.submit({"player_0": 10, "player_1": 20})
    assert r2.done is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/envs/symbolic/test_beauty_contest.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/envs/symbolic/base.py`**

```python
"""Base class for simultaneous-move symbolic games."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from testbed.types import Action, RawObs, RenderContext, StepResult


class SymbolicAdapter:
    """All players act every round (simultaneous turn model)."""

    def __init__(self, num_players: int, num_rounds: int) -> None:
        self.num_players = num_players
        self.num_rounds = num_rounds
        self._ids = [f"player_{i}" for i in range(num_players)]
        self.context = RenderContext()

    def reset(self) -> None:
        self.context = RenderContext()

    def agent_ids(self) -> List[str]:
        return list(self._ids)

    def legal_actions(self, agent_id: str) -> Optional[object]:
        return None

    def pending(self) -> List[Tuple[str, RawObs]]:
        return [(pid, self._observation(pid)) for pid in self._ids]

    def close(self) -> Dict[str, float]:
        return dict(self.context.last_rewards)

    # --- subclass hooks ---
    def _observation(self, agent_id: str) -> RawObs:
        raise NotImplementedError

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        raise NotImplementedError
```

- [ ] **Step 4: Write `testbed/envs/symbolic/beauty_contest.py`**

```python
"""Keynesian beauty contest: guess 2/3 of the group average."""
from __future__ import annotations

from typing import Dict

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, StepResult


class BeautyContestAdapter(SymbolicAdapter):
    def __init__(self, num_players: int = 3, num_rounds: int = 5,
                 low: int = 0, high: int = 100) -> None:
        super().__init__(num_players=num_players, num_rounds=num_rounds)
        self.low = low
        self.high = high

    def _observation(self, agent_id: str) -> RawObs:
        return {
            "agent_id": agent_id,
            "round_index": self.context.round_index,
            "num_players": self.num_players,
            "low": self.low,
            "high": self.high,
            "history": list(self.context.history),
        }

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        choices = {pid: float(actions[pid]) for pid in self._ids}
        mean = sum(choices.values()) / self.num_players
        target = (2.0 / 3.0) * mean

        best = min(abs(v - target) for v in choices.values())
        winners = [pid for pid, v in choices.items() if abs(v - target) == best]
        share = 1.0 / len(winners)
        rewards = {pid: (share if pid in winners else 0.0) for pid in self._ids}

        self.context.round_index += 1
        self.context.last_rewards = rewards
        self.context.history.append({
            "round": self.context.round_index,
            "choices": choices,
            "mean": mean,
            "target": target,
            "winners": winners,
        })
        done = self.context.round_index >= self.num_rounds
        return StepResult(rewards=rewards, done=done,
                          info={"mean": mean, "target": target, "winners": winners})
```

- [ ] **Step 5: Create `tests/envs/symbolic/__init__.py`** (empty), run tests**

Run: `python -m pytest tests/envs/symbolic/test_beauty_contest.py -q`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add testbed/envs/symbolic tests/envs/symbolic
git commit -m "feat: beauty contest adapter (simultaneous symbolic game)"
```

---

## Task 5: GBS (group binary search) adapter

Rules (this project's concrete definition): a hidden integer `target` in `[low, high]`. Each round all players simultaneously submit a guess. After each round the group is told the **median** guess and whether the median is higher or lower than the target (the binary-search hint). Per-round reward is 1.0 for an exact guess, else 0.0. The game ends when the group median equals the target (group converged) or after `num_rounds`.

**Files:**
- Create: `testbed/envs/symbolic/gbs.py`
- Test: `tests/envs/symbolic/test_gbs.py`

- [ ] **Step 1: Write the failing test**

```python
from testbed.envs.symbolic.gbs import GBSAdapter


def test_feedback_direction_and_reward():
    env = GBSAdapter(num_players=3, num_rounds=5, target=50, low=0, high=100)
    env.reset()
    res = env.submit({"player_0": 10, "player_1": 20, "player_2": 30})  # median 20 < 50
    assert res.info["median"] == 20
    assert res.info["direction"] == "higher"  # target is higher than median
    assert res.rewards == {"player_0": 0.0, "player_1": 0.0, "player_2": 0.0}
    assert res.done is False


def test_exact_guess_rewarded():
    env = GBSAdapter(num_players=3, num_rounds=5, target=50, low=0, high=100)
    env.reset()
    res = env.submit({"player_0": 50, "player_1": 10, "player_2": 90})
    assert res.rewards["player_0"] == 1.0
    assert res.rewards["player_1"] == 0.0


def test_group_convergence_ends_game():
    env = GBSAdapter(num_players=3, num_rounds=9, target=50, low=0, high=100)
    env.reset()
    res = env.submit({"player_0": 50, "player_1": 50, "player_2": 50})  # median 50 == target
    assert res.info["direction"] == "correct"
    assert res.done is True


def test_max_rounds_ends_game():
    env = GBSAdapter(num_players=2, num_rounds=1, target=50, low=0, high=100)
    env.reset()
    res = env.submit({"player_0": 10, "player_1": 20})
    assert res.done is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/envs/symbolic/test_gbs.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/envs/symbolic/gbs.py`**

```python
"""Group binary search: players collectively converge on a hidden integer."""
from __future__ import annotations

import statistics
from typing import Dict, Optional

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, StepResult


class GBSAdapter(SymbolicAdapter):
    def __init__(self, num_players: int = 3, num_rounds: int = 9,
                 target: Optional[int] = None, low: int = 0, high: int = 100,
                 seed: int = 0) -> None:
        super().__init__(num_players=num_players, num_rounds=num_rounds)
        self.low = low
        self.high = high
        if target is None:
            import random
            target = random.Random(seed).randint(low, high)
        self.target = target

    def _observation(self, agent_id: str) -> RawObs:
        return {
            "agent_id": agent_id,
            "round_index": self.context.round_index,
            "num_players": self.num_players,
            "low": self.low,
            "high": self.high,
            "history": list(self.context.history),
        }

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        guesses = {pid: int(actions[pid]) for pid in self._ids}
        median = int(statistics.median(guesses.values()))
        if median < self.target:
            direction = "higher"   # target is higher than the median
        elif median > self.target:
            direction = "lower"
        else:
            direction = "correct"

        rewards = {pid: (1.0 if g == self.target else 0.0) for pid, g in guesses.items()}

        self.context.round_index += 1
        self.context.last_rewards = rewards
        self.context.history.append({
            "round": self.context.round_index,
            "guesses": guesses,
            "median": median,
            "direction": direction,
        })
        done = direction == "correct" or self.context.round_index >= self.num_rounds
        return StepResult(rewards=rewards, done=done,
                          info={"median": median, "direction": direction})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/envs/symbolic/test_gbs.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add testbed/envs/symbolic/gbs.py tests/envs/symbolic/test_gbs.py
git commit -m "feat: group binary search adapter"
```

---

## Task 6: Renderer protocol + symbolic renderers

**Files:**
- Create: `testbed/renderers/base.py`, `testbed/renderers/symbolic/beauty_contest.py`, `testbed/renderers/symbolic/gbs.py`
- Test: `tests/renderers/test_symbolic_renderers.py` (create `tests/renderers/__init__.py`)

- [ ] **Step 1: Write the failing test**

```python
from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter
from testbed.envs.symbolic.gbs import GBSAdapter
from testbed.renderers.symbolic.beauty_contest import BeautyContestRenderer
from testbed.renderers.symbolic.gbs import GBSRenderer


def test_beauty_contest_prompt_mentions_rules_and_range():
    env = BeautyContestAdapter(num_players=4, num_rounds=3)
    env.reset()
    obs = env.pending()[0][1]
    r = BeautyContestRenderer()
    sys = r.system_prompt("player_0")
    user = r.render(obs, "player_0", env.context)
    assert "2/3" in sys
    assert "0" in user and "100" in user
    assert "round" in user.lower()


def test_beauty_contest_prompt_includes_history_after_a_round():
    env = BeautyContestAdapter(num_players=2, num_rounds=3)
    env.reset()
    env.submit({"player_0": 10, "player_1": 80})
    obs = env.pending()[0][1]
    user = BeautyContestRenderer().render(obs, "player_0", env.context)
    assert "target" in user.lower()


def test_gbs_prompt_shows_last_direction():
    env = GBSAdapter(num_players=3, num_rounds=9, target=50)
    env.reset()
    env.submit({"player_0": 10, "player_1": 20, "player_2": 30})
    obs = env.pending()[0][1]
    user = GBSRenderer().render(obs, "player_0", env.context)
    assert "higher" in user.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/renderers/test_symbolic_renderers.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/renderers/base.py`**

```python
"""TextRenderer protocol: turn raw observations into prompt text."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from testbed.types import RawObs, RenderContext


@runtime_checkable
class TextRenderer(Protocol):
    def system_prompt(self, agent_id: str) -> str: ...

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str: ...
```

- [ ] **Step 4: Write `testbed/renderers/symbolic/beauty_contest.py`**

```python
from __future__ import annotations

from testbed.types import RawObs, RenderContext


class BeautyContestRenderer:
    def system_prompt(self, agent_id: str) -> str:
        return (
            f"You are {agent_id} in a multi-player Keynesian beauty contest. "
            "Each round, every player picks an integer. The winning number is "
            "2/3 of the average of all picks. The player closest to that winning "
            "number wins the round. Reason about what others will pick, then "
            "respond with your chosen integer."
        )

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        low, high = raw_obs["low"], raw_obs["high"]
        rnd = raw_obs["round_index"] + 1
        lines = [
            f"Round {rnd}. There are {raw_obs['num_players']} players.",
            f"Choose an integer between {low} and {high} (inclusive).",
        ]
        if raw_obs["history"]:
            last = raw_obs["history"][-1]
            lines.append(
                f"Last round the average was {last['mean']:.2f} and the winning "
                f"target (2/3 of average) was {last['target']:.2f}."
            )
        lines.append("Respond with your integer choice in the form: CHOICE: <number>")
        return "\n".join(lines)
```

- [ ] **Step 5: Write `testbed/renderers/symbolic/gbs.py`**

```python
from __future__ import annotations

from testbed.types import RawObs, RenderContext


class GBSRenderer:
    def system_prompt(self, agent_id: str) -> str:
        return (
            f"You are {agent_id} in a cooperative group binary search game. "
            "There is a hidden target integer. Each round all players guess. "
            "After each round you learn the group's median guess and whether the "
            "target is higher or lower than that median. Work with the group to "
            "converge on the target."
        )

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        low, high = raw_obs["low"], raw_obs["high"]
        rnd = raw_obs["round_index"] + 1
        lines = [
            f"Round {rnd}. Hidden target is between {low} and {high} (inclusive).",
        ]
        if raw_obs["history"]:
            last = raw_obs["history"][-1]
            lines.append(
                f"Last round the group median was {last['median']} and the target "
                f"is {last['direction']} than that."
            )
        lines.append("Respond with your integer guess in the form: GUESS: <number>")
        return "\n".join(lines)
```

- [ ] **Step 6: Create `tests/renderers/__init__.py`** (empty), run tests**

Run: `python -m pytest tests/renderers/test_symbolic_renderers.py -q`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add testbed/renderers tests/renderers
git commit -m "feat: renderer protocol + symbolic renderers"
```

---

## Task 7: Parser protocol + symbolic parsers

**Files:**
- Create: `testbed/parsers/base.py`, `testbed/parsers/symbolic/beauty_contest.py`, `testbed/parsers/symbolic/gbs.py`
- Test: `tests/parsers/test_symbolic_parsers.py` (create `tests/parsers/__init__.py`)

- [ ] **Step 1: Write the failing test**

```python
from testbed.types import ParsedAction, ParseError, RenderContext
from testbed.parsers.symbolic.beauty_contest import BeautyContestParser
from testbed.parsers.symbolic.gbs import GBSParser


def _obs(low=0, high=100):
    return {"low": low, "high": high}


def test_beauty_parser_extracts_number():
    p = BeautyContestParser()
    res = p.parse("I think 33 is smart. CHOICE: 33", _obs(), "player_0", RenderContext())
    assert isinstance(res, ParsedAction)
    assert res.value == 33


def test_beauty_parser_plain_number_fallback():
    p = BeautyContestParser()
    res = p.parse("42", _obs(), "player_0", RenderContext())
    assert isinstance(res, ParsedAction)
    assert res.value == 42


def test_beauty_parser_out_of_range_is_error():
    p = BeautyContestParser()
    res = p.parse("CHOICE: 250", _obs(), "player_0", RenderContext())
    assert isinstance(res, ParseError)
    assert "between" in res.feedback.lower()


def test_beauty_parser_no_number_is_error():
    p = BeautyContestParser()
    res = p.parse("I refuse to answer", _obs(), "player_0", RenderContext())
    assert isinstance(res, ParseError)


def test_gbs_parser_extracts_guess():
    p = GBSParser()
    res = p.parse("GUESS: 50", _obs(), "player_0", RenderContext())
    assert isinstance(res, ParsedAction)
    assert res.value == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/parsers/test_symbolic_parsers.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/parsers/base.py`**

```python
"""ActionParser protocol + a shared integer-extraction helper."""
from __future__ import annotations

import re
from typing import Optional, Protocol, runtime_checkable

from testbed.types import ParseResult, RawObs, RenderContext


@runtime_checkable
class ActionParser(Protocol):
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult: ...


def extract_int(text: str, keyword: Optional[str] = None) -> Optional[int]:
    """Extract an integer. If keyword given (e.g. 'CHOICE'), prefer 'CHOICE: <n>'."""
    if keyword:
        m = re.search(rf"{keyword}\s*[:=]\s*(-?\d+)", text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    nums = re.findall(r"-?\d+", text)
    if nums:
        return int(nums[-1])  # last number is usually the final answer
    return None
```

- [ ] **Step 4: Write `testbed/parsers/symbolic/beauty_contest.py`**

```python
from __future__ import annotations

from testbed.parsers.base import extract_int
from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class BeautyContestParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        low, high = raw_obs["low"], raw_obs["high"]
        n = extract_int(completion, keyword="CHOICE")
        if n is None:
            return ParseError(
                feedback=("I could not find a number in your reply. Respond with "
                          "'CHOICE: <number>'.")
            )
        if n < low or n > high:
            return ParseError(
                feedback=(f"Your choice {n} is out of range. Pick an integer "
                          f"between {low} and {high}. Respond with 'CHOICE: <number>'.")
            )
        return ParsedAction(value=n)
```

- [ ] **Step 5: Write `testbed/parsers/symbolic/gbs.py`**

```python
from __future__ import annotations

from testbed.parsers.base import extract_int
from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class GBSParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        low, high = raw_obs["low"], raw_obs["high"]
        n = extract_int(completion, keyword="GUESS")
        if n is None:
            return ParseError(
                feedback="I could not find a number. Respond with 'GUESS: <number>'."
            )
        if n < low or n > high:
            return ParseError(
                feedback=(f"Your guess {n} is out of range. Pick an integer between "
                          f"{low} and {high}. Respond with 'GUESS: <number>'.")
            )
        return ParsedAction(value=n)
```

- [ ] **Step 6: Create `tests/parsers/__init__.py`** (empty), run tests**

Run: `python -m pytest tests/parsers/test_symbolic_parsers.py -q`
Expected: PASS (5 passed)

- [ ] **Step 7: Commit**

```bash
git add testbed/parsers tests/parsers
git commit -m "feat: parser protocol + symbolic parsers with error feedback"
```

---

## Task 8: Steering — protocol, no-op, prompt injection

**Files:**
- Create: `testbed/steering/base.py`, `testbed/steering/noop.py`, `testbed/steering/prompt_injection.py`
- Test: `tests/steering/test_prompt_steering.py` (create `tests/steering/__init__.py`)

- [ ] **Step 1: Write the failing test**

```python
from testbed.steering.noop import NoOpSteering
from testbed.steering.prompt_injection import PromptInjectionSteering


def test_noop_is_identity():
    s = NoOpSteering()
    sys, user = s.apply_to_prompt("SYS", "USER", "player_0")
    assert sys == "SYS" and user == "USER"
    assert s.steering_spec("player_0") is None


def test_prompt_injection_appends_per_agent_suffix():
    s = PromptInjectionSteering(per_agent={
        "player_0": {"system_suffix": " Be ruthlessly competitive."},
        "player_1": {"user_prefix": "Remember to cooperate. "},
    })
    sys0, user0 = s.apply_to_prompt("SYS", "USER", "player_0")
    assert sys0.endswith("Be ruthlessly competitive.")
    assert user0 == "USER"
    sys1, user1 = s.apply_to_prompt("SYS", "USER", "player_1")
    assert user1.startswith("Remember to cooperate.")


def test_prompt_injection_unconfigured_agent_unchanged():
    s = PromptInjectionSteering(per_agent={})
    sys, user = s.apply_to_prompt("SYS", "USER", "player_9")
    assert sys == "SYS" and user == "USER"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/steering/test_prompt_steering.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/steering/base.py`**

```python
"""SteeringMethod protocol."""
from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable

from testbed.types import SteeringSpec


@runtime_checkable
class SteeringMethod(Protocol):
    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> Tuple[str, str]: ...

    def steering_spec(self, agent_id: str) -> Optional[SteeringSpec]: ...
```

- [ ] **Step 4: Write `testbed/steering/noop.py`**

```python
from __future__ import annotations

from typing import Optional, Tuple

from testbed.types import SteeringSpec


class NoOpSteering:
    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> Tuple[str, str]:
        return system_prompt, user_prompt

    def steering_spec(self, agent_id: str) -> Optional[SteeringSpec]:
        return None
```

- [ ] **Step 5: Write `testbed/steering/prompt_injection.py`**

```python
from __future__ import annotations

from typing import Dict, Optional, Tuple

from testbed.types import SteeringSpec


class PromptInjectionSteering:
    """Per-agent prompt edits. Config: {agent_id: {system_suffix?, user_prefix?}}."""

    def __init__(self, per_agent: Dict[str, Dict[str, str]]) -> None:
        self.per_agent = per_agent

    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> Tuple[str, str]:
        cfg = self.per_agent.get(agent_id, {})
        if "system_suffix" in cfg:
            system_prompt = system_prompt + cfg["system_suffix"]
        if "user_prefix" in cfg:
            user_prompt = cfg["user_prefix"] + user_prompt
        return system_prompt, user_prompt

    def steering_spec(self, agent_id: str) -> Optional[SteeringSpec]:
        return None  # prompt injection works purely on text
```

- [ ] **Step 6: Create `tests/steering/__init__.py`** (empty), run tests**

Run: `python -m pytest tests/steering/test_prompt_steering.py -q`
Expected: PASS (3 passed)

- [ ] **Step 7: Commit**

```bash
git add testbed/steering tests/steering
git commit -m "feat: steering protocol, no-op, prompt injection"
```

---

## Task 9: Activation steering — vector load + forward-hook factory (CPU toy test)

This is the core steering mechanism. `ActivationSteering` loads a vector and provides a hook function that adds `coefficient * vector` to a module's output. We verify the hook math on a tiny CPU `nn.Module` (no Qwen needed), proving correctness here.

**Files:**
- Create: `testbed/steering/activation.py`
- Test: `tests/steering/test_activation_steering.py`

- [ ] **Step 1: Write the failing test** (requires torch, which is present on CPU)

```python
import numpy as np
import torch
import torch.nn as nn

from testbed.steering.activation import ActivationSteering, make_steering_hook


def test_vector_loaded_from_npy(tmp_path):
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    p = tmp_path / "vec.npy"
    np.save(p, v)
    s = ActivationSteering(per_agent={
        "player_0": {"layer": "l", "vector_path": str(p), "coefficient": 2.0},
    })
    spec = s.steering_spec("player_0")
    assert spec is not None and spec.layer == "l" and spec.coefficient == 2.0
    vec = s.load_vector("player_0")
    assert torch.allclose(vec, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_hook_adds_scaled_vector_to_output():
    hidden = 4
    vec = torch.ones(hidden)
    hook = make_steering_hook(vec, coefficient=3.0)

    layer = nn.Identity()
    handle = layer.register_forward_hook(hook)
    x = torch.zeros((1, 2, hidden))  # (batch, seq, hidden)
    out = layer(x)
    handle.remove()
    # every position should have +3.0 added
    assert torch.allclose(out, torch.full((1, 2, hidden), 3.0))


def test_hook_handles_tuple_output():
    hidden = 3
    vec = torch.tensor([1.0, 0.0, 0.0])
    hook = make_steering_hook(vec, coefficient=5.0)

    class TupleLayer(nn.Module):
        def forward(self, x):
            return (x, "aux")  # transformer blocks often return tuples

    layer = TupleLayer()
    handle = layer.register_forward_hook(hook)
    x = torch.zeros((1, 1, hidden))
    out = layer(x)
    handle.remove()
    assert isinstance(out, tuple)
    assert torch.allclose(out[0], torch.tensor([[[5.0, 0.0, 0.0]]]))
    assert out[1] == "aux"


def test_unconfigured_agent_has_no_spec():
    s = ActivationSteering(per_agent={})
    assert s.steering_spec("player_3") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/steering/test_activation_steering.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/steering/activation.py`**

```python
"""Activation steering: add a steering vector into the residual stream via a hook.

The apply-path is fully implemented and unit-tested here on a CPU toy module.
Only the offline derivation of *meaningful* vectors is out of scope; drop a
precomputed .pt/.npy vector and point a SteeringSpec at it.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from testbed.types import SteeringSpec


def make_steering_hook(vector, coefficient: float) -> Callable:
    """Return a forward hook that adds coefficient*vector to a module's output.

    Handles modules whose output is a Tensor or a tuple whose first element is the
    hidden-state Tensor (as in most HF decoder blocks).
    """
    import torch

    vec = vector if hasattr(vector, "to") else torch.tensor(vector)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + coefficient * vec.to(hidden.dtype).to(hidden.device)
            return (hidden,) + tuple(output[1:])
        hidden = output
        return hidden + coefficient * vec.to(hidden.dtype).to(hidden.device)

    return hook


class ActivationSteering:
    """Per-agent activation steering config: {agent_id: {layer, vector_path, coefficient}}."""

    def __init__(self, per_agent: Dict[str, Dict]) -> None:
        self.per_agent = per_agent
        self._cache: Dict[str, object] = {}

    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> Tuple[str, str]:
        return system_prompt, user_prompt  # activation steering does not touch text

    def steering_spec(self, agent_id: str) -> Optional[SteeringSpec]:
        cfg = self.per_agent.get(agent_id)
        if cfg is None:
            return None
        return SteeringSpec(
            method="activation",
            layer=cfg["layer"],
            vector_path=cfg["vector_path"],
            coefficient=float(cfg.get("coefficient", 1.0)),
        )

    def load_vector(self, agent_id: str):
        import numpy as np
        import torch

        if agent_id in self._cache:
            return self._cache[agent_id]
        spec = self.steering_spec(agent_id)
        if spec is None:
            raise KeyError(f"No activation steering configured for {agent_id}")
        path = spec.vector_path
        if path.endswith(".npy"):
            vec = torch.tensor(np.load(path))
        else:
            vec = torch.load(path)
        vec = vec.float()
        self._cache[agent_id] = vec
        return vec
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/steering/test_activation_steering.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add testbed/steering/activation.py tests/steering/test_activation_steering.py
git commit -m "feat: activation steering vector load + forward-hook factory (CPU-tested)"
```

---

## Task 10: Policy protocol + StubPolicy

**Files:**
- Create: `testbed/policy/base.py`
- Test: `tests/policy/test_stub_policy.py` (create `tests/policy/__init__.py`)

- [ ] **Step 1: Write the failing test**

```python
from testbed.policy.base import StubPolicy


def test_stub_policy_returns_scripted_completions():
    p = StubPolicy(scripted={"player_0": ["CHOICE: 33", "CHOICE: 22"]})
    assert p.act("s", "u", "player_0", None) == "CHOICE: 33"
    assert p.act("s", "u", "player_0", None) == "CHOICE: 22"


def test_stub_policy_default_when_unscripted():
    p = StubPolicy(scripted={}, default="CHOICE: 0")
    assert p.act("s", "u", "player_5", None) == "CHOICE: 0"


def test_stub_policy_records_calls():
    p = StubPolicy(scripted={}, default="x")
    p.act("SYS", "USER", "player_0", None)
    assert p.calls[-1]["agent_id"] == "player_0"
    assert p.calls[-1]["user"] == "USER"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/policy/test_stub_policy.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/policy/base.py`**

```python
"""Policy protocol + a StubPolicy for GPU-free testing."""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol, runtime_checkable

from testbed.types import SteeringSpec


@runtime_checkable
class Policy(Protocol):
    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec]) -> str: ...


class StubPolicy:
    """Returns scripted completions per agent; records calls. No model required."""

    def __init__(self, scripted: Dict[str, List[str]], default: str = "CHOICE: 0") -> None:
        self.scripted = {k: list(v) for k, v in scripted.items()}
        self.default = default
        self.calls: List[Dict[str, str]] = []

    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec]) -> str:
        self.calls.append({"agent_id": agent_id, "system": system_prompt,
                           "user": user_prompt})
        queue = self.scripted.get(agent_id)
        if queue:
            return queue.pop(0)
        return self.default
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/policy/test_stub_policy.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add testbed/policy/base.py tests/policy/__init__.py tests/policy/test_stub_policy.py
git commit -m "feat: policy protocol + stub policy"
```

---

## Task 11: Logger

**Files:**
- Create: `testbed/logging_/episode_logger.py`
- Test: `tests/test_logger.py`

- [ ] **Step 1: Write the failing test**

```python
import json
from testbed.logging_.episode_logger import EpisodeLogger


def test_logger_writes_jsonl_lines(tmp_path):
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="run1", episode=0)
    logger.log_step(game="beauty_contest", turn=0, agent_id="player_0",
                    system_prompt="s", user_prompt="u", completion="CHOICE: 33",
                    parsed_action=33, parse_retries=0, reward=0.0,
                    steering_spec_id="noop")
    logger.log_step(game="beauty_contest", turn=0, agent_id="player_1",
                    system_prompt="s", user_prompt="u", completion="CHOICE: 50",
                    parsed_action=50, parse_retries=1, reward=1.0,
                    steering_spec_id="noop")
    logger.close(summary={"winner": "player_1"})

    path = tmp_path / "run1" / "episode_0.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["agent_id"] == "player_0"
    assert rec["parsed_action"] == 33

    summary = json.loads((tmp_path / "run1" / "episode_0.summary.json").read_text(encoding="utf-8"))
    assert summary["winner"] == "player_1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_logger.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/logging_/episode_logger.py`**

```python
"""JSONL trace logger: one line per agent per turn, plus an episode summary."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class EpisodeLogger:
    def __init__(self, run_dir: str, run_id: str, episode: int) -> None:
        self.dir = os.path.join(run_dir, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.run_id = run_id
        self.episode = episode
        self.path = os.path.join(self.dir, f"episode_{episode}.jsonl")
        self._fh = open(self.path, "w", encoding="utf-8")

    def log_step(self, *, game: str, turn: int, agent_id: str, system_prompt: str,
                 user_prompt: str, completion: str, parsed_action: Any,
                 parse_retries: int, reward: float,
                 steering_spec_id: Optional[str]) -> None:
        rec = {
            "run_id": self.run_id, "episode": self.episode, "game": game,
            "turn": turn, "agent_id": agent_id, "system_prompt": system_prompt,
            "user_prompt": user_prompt, "completion": completion,
            "parsed_action": parsed_action, "parse_retries": parse_retries,
            "reward": reward, "steering_spec_id": steering_spec_id,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if not self._fh.closed:
            self._fh.close()
        if summary is not None:
            spath = os.path.join(self.dir, f"episode_{self.episode}.summary.json")
            with open(spath, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_logger.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add testbed/logging_/episode_logger.py tests/test_logger.py
git commit -m "feat: episode JSONL logger + summary"
```

---

## Task 12: Orchestrator

The single game loop. Iterates `pending()`, renders, applies steering, calls policy, parses with bounded retries (re-prompting with parser feedback appended), logs, then `submit()`s. Works for both turn models because it just trusts `pending()`.

**Files:**
- Create: `testbed/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test** (full loop, no GPU — uses StubPolicy + beauty contest)

```python
from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter
from testbed.renderers.symbolic.beauty_contest import BeautyContestRenderer
from testbed.parsers.symbolic.beauty_contest import BeautyContestParser
from testbed.policy.base import StubPolicy
from testbed.steering.noop import NoOpSteering
from testbed.logging_.episode_logger import EpisodeLogger
from testbed.orchestrator import Orchestrator


def test_full_episode_runs_and_logs(tmp_path):
    env = BeautyContestAdapter(num_players=2, num_rounds=2)
    policy = StubPolicy(scripted={
        "player_0": ["CHOICE: 10", "CHOICE: 10"],
        "player_1": ["CHOICE: 20", "CHOICE: 20"],
    })
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="r", episode=0)
    orch = Orchestrator(
        env=env, renderer=BeautyContestRenderer(), parser=BeautyContestParser(),
        policy=policy, steering=NoOpSteering(), logger=logger,
        game="beauty_contest", max_parse_retries=3,
    )
    final = orch.run_episode()
    assert set(final.keys()) == {"player_0", "player_1"}
    # 2 rounds x 2 players = 4 policy calls
    assert len(policy.calls) == 4


def test_orchestrator_reprompts_on_parse_error(tmp_path):
    env = BeautyContestAdapter(num_players=1, num_rounds=1)
    # first completion is junk, second is valid -> one retry
    policy = StubPolicy(scripted={"player_0": ["no number here", "CHOICE: 30"]})
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="r2", episode=0)
    orch = Orchestrator(
        env=env, renderer=BeautyContestRenderer(), parser=BeautyContestParser(),
        policy=policy, steering=NoOpSteering(), logger=logger,
        game="beauty_contest", max_parse_retries=3,
    )
    orch.run_episode()
    assert len(policy.calls) == 2  # initial + 1 retry


def test_orchestrator_falls_back_after_max_retries(tmp_path):
    env = BeautyContestAdapter(num_players=1, num_rounds=1)
    policy = StubPolicy(scripted={}, default="garbage no digits zzz")
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="r3", episode=0)
    orch = Orchestrator(
        env=env, renderer=BeautyContestRenderer(), parser=BeautyContestParser(),
        policy=policy, steering=NoOpSteering(), logger=logger,
        game="beauty_contest", max_parse_retries=2, fallback_action=0,
    )
    final = orch.run_episode()
    # initial + 2 retries = 3 calls, then fallback used
    assert len(policy.calls) == 3
    assert final["player_0"] == 1.0  # only player, wins by default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_orchestrator.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/orchestrator.py`**

```python
"""The single, game-agnostic episode loop."""
from __future__ import annotations

from typing import Any, Dict, Optional

from testbed.types import ParseError, ParsedAction


class Orchestrator:
    def __init__(self, *, env, renderer, parser, policy, steering, logger,
                 game: str, max_parse_retries: int = 5,
                 fallback_action: Any = 0) -> None:
        self.env = env
        self.renderer = renderer
        self.parser = parser
        self.policy = policy
        self.steering = steering
        self.logger = logger
        self.game = game
        self.max_parse_retries = max_parse_retries
        self.fallback_action = fallback_action

    def _act_one(self, agent_id: str, raw_obs, turn: int):
        system = self.renderer.system_prompt(agent_id)
        base_user = self.renderer.render(raw_obs, agent_id, self.env.context
                                         if hasattr(self.env, "context") else None)
        system, base_user = self.steering.apply_to_prompt(system, base_user, agent_id)
        spec = self.steering.steering_spec(agent_id)
        spec_id = spec.method if spec is not None else "noop"

        user = base_user
        retries = 0
        completion = ""
        while True:
            completion = self.policy.act(system, user, agent_id, spec)
            result = self.parser.parse(completion, raw_obs, agent_id,
                                       self.env.context if hasattr(self.env, "context") else None)
            if isinstance(result, ParsedAction):
                action = result.value
                break
            if retries >= self.max_parse_retries:
                action = self.fallback_action
                break
            retries += 1
            user = base_user + "\n\n" + result.feedback  # re-prompt with feedback

        return action, completion, retries, spec_id

    def run_episode(self) -> Dict[str, float]:
        self.env.reset()
        turn = 0
        last_rewards: Dict[str, float] = {}
        done = False
        while not done:
            pending = self.env.pending()
            actions: Dict[str, Any] = {}
            decided = {}
            for agent_id, raw_obs in pending:
                action, completion, retries, spec_id = self._act_one(agent_id, raw_obs, turn)
                actions[agent_id] = action
                decided[agent_id] = (raw_obs, completion, retries, spec_id)

            result = self.env.submit(actions)
            last_rewards = result.rewards
            for agent_id, (raw_obs, completion, retries, spec_id) in decided.items():
                system = self.renderer.system_prompt(agent_id)
                self.logger.log_step(
                    game=self.game, turn=turn, agent_id=agent_id,
                    system_prompt=system, user_prompt="",  # full user logged via completion context
                    completion=completion, parsed_action=actions[agent_id],
                    parse_retries=retries, reward=result.rewards.get(agent_id, 0.0),
                    steering_spec_id=spec_id,
                )
            turn += 1
            done = result.done

        self.logger.close(summary={"final_rewards": last_rewards})
        return last_rewards
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_orchestrator.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add testbed/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: game-agnostic orchestrator with parse-retry feedback loop"
```

---

## Task 13: TextArena adapter + renderer + parser

TextArena is turn-based and text-native. The adapter wraps `ta.make`; `pending()` returns the single current player from `get_observation()`; `submit()` forwards the action string and advances. Renderer/parser are pass-through. We test adapter *logic* against a fake env (no lib needed); a real-lib smoke test is gated with `importorskip`.

**Files:**
- Create: `testbed/envs/textarena/ta_adapter.py`, `testbed/renderers/textarena.py`, `testbed/parsers/textarena.py`
- Test: `tests/envs/textarena/test_ta_adapter.py` (create `tests/envs/textarena/__init__.py`)

- [ ] **Step 1: Write the failing test** (fake TextArena env injected)

```python
import pytest

from testbed.envs.textarena.ta_adapter import TextArenaAdapter
from testbed.renderers.textarena import TextArenaRenderer
from testbed.parsers.textarena import TextArenaParser
from testbed.types import ParsedAction, RenderContext


class FakeTAEnv:
    """Minimal stand-in for a textarena env: 2 turns then done."""
    def __init__(self):
        self._turn = 0
        self.stepped = []
    def reset(self, num_players): self.num_players = num_players; self._turn = 0
    def get_observation(self):
        pid = self._turn % self.num_players
        return pid, f"observation for player {pid}"
    def step(self, action):
        self.stepped.append(action)
        self._turn += 1
        done = self._turn >= 2
        return done, {"info": self._turn}
    def close(self):
        return {0: 1.0, 1: -1.0}, {"game": "fake"}


def test_pending_returns_single_current_player():
    env = TextArenaAdapter(env_id="Fake-v0", num_players=2, _env=FakeTAEnv())
    env.reset()
    pend = env.pending()
    assert len(pend) == 1
    assert pend[0][0] == "player_0"
    assert "player 0" in pend[0][1]


def test_submit_forwards_action_and_advances():
    fake = FakeTAEnv()
    env = TextArenaAdapter(env_id="Fake-v0", num_players=2, _env=fake)
    env.reset()
    env.pending()
    res = env.submit({"player_0": "[Vote] 1"})
    assert fake.stepped == ["[Vote] 1"]
    assert res.done is False
    env.pending()
    res2 = env.submit({"player_1": "ok"})
    assert res2.done is True


def test_close_maps_rewards_to_player_ids():
    env = TextArenaAdapter(env_id="Fake-v0", num_players=2, _env=FakeTAEnv())
    env.reset()
    rewards = env.close()
    assert rewards["player_0"] == 1.0
    assert rewards["player_1"] == -1.0


def test_renderer_is_passthrough():
    r = TextArenaRenderer()
    assert r.render("raw obs string", "player_0", RenderContext()) == "raw obs string"


def test_parser_passes_completion_through():
    p = TextArenaParser()
    res = p.parse("  [Vote] Player 3  ", "obs", "player_0", RenderContext())
    assert isinstance(res, ParsedAction)
    assert res.value == "[Vote] Player 3"


@pytest.mark.gpu
def test_real_textarena_smoke():
    ta = pytest.importorskip("textarena")
    env = TextArenaAdapter(env_id="ThreePlayerTicTacToe-v0", num_players=3)
    env.reset()
    pend = env.pending()
    assert len(pend) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/envs/textarena/test_ta_adapter.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Create `tests/envs/textarena/__init__.py`** (empty), write `testbed/envs/textarena/ta_adapter.py`**

```python
"""Adapter for TextArena's turn-based, text-native multiplayer games."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from testbed.types import Action, RawObs, RenderContext, StepResult


class TextArenaAdapter:
    def __init__(self, env_id: str, num_players: int, _env=None) -> None:
        self.env_id = env_id
        self.num_players = num_players
        self._env = _env  # inject for tests; otherwise built in reset()
        self.context = RenderContext()
        self._current_pid: Optional[int] = None

    def _pid_str(self, pid: int) -> str:
        return f"player_{pid}"

    def reset(self) -> None:
        if self._env is None:
            import textarena as ta
            self._env = ta.make(env_id=self.env_id)
        self._env.reset(num_players=self.num_players)
        self.context = RenderContext()
        self._current_pid = None

    def agent_ids(self) -> List[str]:
        return [self._pid_str(i) for i in range(self.num_players)]

    def legal_actions(self, agent_id: str) -> Optional[object]:
        return None  # TextArena validates actions itself

    def pending(self) -> List[Tuple[str, RawObs]]:
        pid, observation = self._env.get_observation()
        self._current_pid = pid
        return [(self._pid_str(pid), observation)]

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        agent_id = self._pid_str(self._current_pid)
        action = actions[agent_id]
        done, info = self._env.step(action)
        rewards: Dict[str, float] = {}
        if done:
            ta_rewards, close_info = self._env.close()
            rewards = {self._pid_str(k): float(v) for k, v in ta_rewards.items()}
            self.context.last_rewards = rewards
            info = {**(info or {}), "close_info": close_info}
        return StepResult(rewards=rewards, done=done, info=info or {})

    def close(self) -> Dict[str, float]:
        if self.context.last_rewards:
            return dict(self.context.last_rewards)
        ta_rewards, _ = self._env.close()
        return {self._pid_str(k): float(v) for k, v in ta_rewards.items()}
```

- [ ] **Step 4: Write `testbed/renderers/textarena.py`**

```python
"""Pass-through renderer: TextArena already produces the observation string."""
from __future__ import annotations

from testbed.types import RawObs, RenderContext


class TextArenaRenderer:
    def __init__(self, system_prompt_text: str = "") -> None:
        self._sys = system_prompt_text

    def system_prompt(self, agent_id: str) -> str:
        return self._sys

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        return raw_obs  # already a string
```

- [ ] **Step 5: Write `testbed/parsers/textarena.py`**

```python
"""Near pass-through parser: TextArena validates actions itself."""
from __future__ import annotations

from testbed.types import ParsedAction, ParseError, ParseResult, RawObs, RenderContext


class TextArenaParser:
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: RenderContext) -> ParseResult:
        text = (completion or "").strip()
        if not text:
            return ParseError(feedback="Empty response. Please provide an action.")
        return ParsedAction(value=text)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/envs/textarena/test_ta_adapter.py -q`
Expected: PASS (5 passed, 1 skipped — the `gpu`/importorskip smoke test)

- [ ] **Step 7: Commit**

```bash
git add testbed/envs/textarena testbed/renderers/textarena.py testbed/parsers/textarena.py tests/envs/textarena
git commit -m "feat: TextArena adapter (turn-based) + passthrough renderer/parser"
```

---

## Task 14: Registry

Maps a game spec `{family, id}` to its `(adapter, renderer, parser)` triple.

**Files:**
- Create: `testbed/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
from testbed.registry import build_game
from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter
from testbed.renderers.symbolic.beauty_contest import BeautyContestRenderer
from testbed.parsers.symbolic.beauty_contest import BeautyContestParser
from testbed.envs.symbolic.gbs import GBSAdapter
from testbed.envs.textarena.ta_adapter import TextArenaAdapter


def test_build_beauty_contest():
    env, renderer, parser = build_game(
        family="symbolic", game_id="beauty_contest",
        num_players=3, env_kwargs={"num_rounds": 4})
    assert isinstance(env, BeautyContestAdapter)
    assert isinstance(renderer, BeautyContestRenderer)
    assert isinstance(parser, BeautyContestParser)
    assert env.num_players == 3
    assert env.num_rounds == 4


def test_build_gbs():
    env, renderer, parser = build_game(
        family="symbolic", game_id="gbs", num_players=5, env_kwargs={})
    assert isinstance(env, GBSAdapter)
    assert env.num_players == 5


def test_build_textarena():
    env, renderer, parser = build_game(
        family="textarena", game_id="SecretMafia-v0", num_players=7, env_kwargs={})
    assert isinstance(env, TextArenaAdapter)
    assert env.env_id == "SecretMafia-v0"
    assert env.num_players == 7


def test_unknown_game_raises():
    import pytest
    with pytest.raises(ValueError):
        build_game(family="symbolic", game_id="nope", num_players=2, env_kwargs={})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_registry.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/registry.py`**

```python
"""Map a game spec to its (adapter, renderer, parser) triple."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from testbed.envs.symbolic.beauty_contest import BeautyContestAdapter
from testbed.envs.symbolic.gbs import GBSAdapter
from testbed.envs.textarena.ta_adapter import TextArenaAdapter
from testbed.parsers.symbolic.beauty_contest import BeautyContestParser
from testbed.parsers.symbolic.gbs import GBSParser
from testbed.parsers.textarena import TextArenaParser
from testbed.renderers.symbolic.beauty_contest import BeautyContestRenderer
from testbed.renderers.symbolic.gbs import GBSRenderer
from testbed.renderers.textarena import TextArenaParser as _unused  # noqa  (keep imports explicit)
from testbed.renderers.textarena import TextArenaRenderer

_SYMBOLIC = {
    "beauty_contest": (BeautyContestAdapter, BeautyContestRenderer, BeautyContestParser),
    "gbs": (GBSAdapter, GBSRenderer, GBSParser),
}


def build_game(*, family: str, game_id: str, num_players: int,
               env_kwargs: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    if family == "symbolic":
        if game_id not in _SYMBOLIC:
            raise ValueError(f"Unknown symbolic game: {game_id}")
        AdapterCls, RendererCls, ParserCls = _SYMBOLIC[game_id]
        env = AdapterCls(num_players=num_players, **env_kwargs)
        return env, RendererCls(), ParserCls()
    if family == "textarena":
        env = TextArenaAdapter(env_id=game_id, num_players=num_players, **env_kwargs)
        return env, TextArenaRenderer(), TextArenaParser()
    raise ValueError(f"Unknown game family: {family}")
```

NOTE: remove the bogus `from testbed.renderers.textarena import TextArenaParser as _unused` line — it is intentionally wrong to flag that `TextArenaParser` lives in `parsers`, not `renderers`. Final import block must be:

```python
from testbed.parsers.textarena import TextArenaParser
from testbed.renderers.textarena import TextArenaRenderer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_registry.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add testbed/registry.py tests/test_registry.py
git commit -m "feat: game registry"
```

---

## Task 15: TransformersPolicy (steering-capable, GPU-gated)

Loads an HF causal LM, builds a chat prompt, and—when a `SteeringSpec` is present—registers a forward hook on the named submodule for the duration of generation, then removes it. Tested behaviorally on CPU with a **tiny fake model** to prove hook registration/removal logic without Qwen.

**Files:**
- Create: `testbed/policy/transformers_policy.py`
- Test: `tests/policy/test_transformers_policy.py`

- [ ] **Step 1: Write the failing test** (hook lifecycle tested via a fake model; real Qwen gated)

```python
import pytest
import torch
import torch.nn as nn

from testbed.policy.transformers_policy import _resolve_submodule, _SteeringSession
from testbed.types import SteeringSpec
from testbed.steering.activation import make_steering_hook


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Identity()


def test_resolve_submodule_by_dotted_name():
    m = TinyModel()
    assert _resolve_submodule(m, "block") is m.block


def test_steering_session_adds_and_removes_hook():
    m = TinyModel()
    vec = torch.ones(3)
    hook = make_steering_hook(vec, coefficient=2.0)
    assert len(m.block._forward_hooks) == 0
    with _SteeringSession(m, "block", hook):
        assert len(m.block._forward_hooks) == 1
        out = m.block(torch.zeros((1, 1, 3)))
        assert torch.allclose(out, torch.full((1, 1, 3), 2.0))
    assert len(m.block._forward_hooks) == 0  # removed on exit


@pytest.mark.gpu
def test_transformers_policy_generates_with_qwen():
    from testbed.policy.transformers_policy import TransformersPolicy
    p = TransformersPolicy(model_id="Qwen/Qwen2.5-3B-Instruct")
    out = p.act("You are a helpful assistant.", "Say the word 'ok'.", "player_0", None)
    assert isinstance(out, str) and len(out) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/policy/test_transformers_policy.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/policy/transformers_policy.py`**

```python
"""In-process HuggingFace policy with activation-steering forward hooks."""
from __future__ import annotations

from typing import Optional

from testbed.steering.activation import make_steering_hook
from testbed.types import SteeringSpec


def _resolve_submodule(model, dotted_name: str):
    """Resolve 'a.b.c' to a submodule of model."""
    obj = model
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return obj


class _SteeringSession:
    """Context manager: register a forward hook on a submodule, remove on exit."""

    def __init__(self, model, layer_name: str, hook):
        self.module = _resolve_submodule(model, layer_name)
        self.hook = hook
        self.handle = None

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self.hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
        return False


class TransformersPolicy:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-3B-Instruct",
                 temperature: float = 0.7, max_new_tokens: int = 256,
                 device: Optional[str] = None,
                 steering: Optional["object"] = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto").to(self.device)
        self.model.eval()
        # the steering method (ActivationSteering) used to load vectors by agent
        self.steering = steering

    def _build_inputs(self, system_prompt: str, user_prompt: str):
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def _generate(self, inputs) -> str:
        import torch
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec]) -> str:
        inputs = self._build_inputs(system_prompt, user_prompt)
        if steering is not None and steering.method == "activation":
            if self.steering is None:
                raise ValueError("Activation steering requested but no steering "
                                 "method bound to the policy to load vectors.")
            vec = self.steering.load_vector(agent_id)
            hook = make_steering_hook(vec, coefficient=steering.coefficient)
            with _SteeringSession(self.model, steering.layer, hook):
                return self._generate(inputs)
        return self._generate(inputs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/policy/test_transformers_policy.py -q`
Expected: PASS (2 passed, 1 skipped — the Qwen gpu test)

- [ ] **Step 5: Commit**

```bash
git add testbed/policy/transformers_policy.py tests/policy/test_transformers_policy.py
git commit -m "feat: transformers policy with activation-steering hook session (CPU-tested lifecycle)"
```

---

## Task 16: VLLMPolicy (fast baseline, server-gated)

OpenAI-compatible client to a local vLLM server. No steering. Logic (message building, response extraction) tested against a fake client; live call gated.

**Files:**
- Create: `testbed/policy/vllm_policy.py`
- Test: `tests/policy/test_vllm_policy.py`

- [ ] **Step 1: Write the failing test** (fake OpenAI-style client injected)

```python
import pytest
from testbed.policy.vllm_policy import VLLMPolicy
from testbed.types import SteeringSpec


class _FakeCompletions:
    def create(self, **kwargs):
        class R:
            choices = [type("C", (), {"message": type("M", (), {"content": "hello"})()})()]
        # capture for assertions
        _FakeCompletions.last_kwargs = kwargs
        return R()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


def test_vllm_policy_returns_content():
    p = VLLMPolicy(model_id="Qwen/Qwen2.5-3B-Instruct", _client=_FakeClient())
    out = p.act("SYS", "USER", "player_0", None)
    assert out == "hello"
    msgs = _FakeCompletions.last_kwargs["messages"]
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "SYS"
    assert msgs[1]["content"] == "USER"


def test_vllm_policy_rejects_activation_steering():
    p = VLLMPolicy(model_id="m", _client=_FakeClient())
    spec = SteeringSpec(method="activation", layer="l", vector_path="v.pt", coefficient=1.0)
    with pytest.raises(ValueError):
        p.act("SYS", "USER", "player_0", spec)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/policy/test_vllm_policy.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/policy/vllm_policy.py`**

```python
"""Fast baseline policy: OpenAI-compatible client to a local vLLM server."""
from __future__ import annotations

from typing import Optional

from testbed.types import SteeringSpec


class VLLMPolicy:
    def __init__(self, model_id: str, endpoint: str = "http://localhost:8000/v1",
                 api_key: str = "EMPTY", temperature: float = 0.7,
                 max_new_tokens: int = 256, _client=None) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        if _client is not None:
            self.client = _client
        else:
            from openai import OpenAI
            self.client = OpenAI(base_url=endpoint, api_key=api_key)

    def act(self, system_prompt: str, user_prompt: str, agent_id: str,
            steering: Optional[SteeringSpec]) -> str:
        if steering is not None and steering.method == "activation":
            raise ValueError("VLLMPolicy cannot apply activation steering; use "
                             "TransformersPolicy for activation runs.")
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=self.temperature, max_tokens=self.max_new_tokens)
        return resp.choices[0].message.content
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/policy/test_vllm_policy.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add testbed/policy/vllm_policy.py tests/policy/test_vllm_policy.py
git commit -m "feat: vLLM client policy (baseline, no steering)"
```

---

## Task 17: Config loader + run_episode CLI + end-to-end wiring

Loads YAML config, builds the game via the registry, builds the steering method and policy, runs an episode, and writes logs.

**Files:**
- Create: `testbed/config.py`, `config/run_config.yaml`, `scripts/run_episode.py`
- Test: `tests/test_config_and_e2e.py`

- [ ] **Step 1: Write the failing test** (end-to-end with stub policy, driven by a config dict)

```python
from testbed.config import build_steering, build_policy, RunConfig
from testbed.registry import build_game
from testbed.policy.base import StubPolicy
from testbed.steering.noop import NoOpSteering
from testbed.steering.prompt_injection import PromptInjectionSteering
from testbed.steering.activation import ActivationSteering
from testbed.logging_.episode_logger import EpisodeLogger
from testbed.orchestrator import Orchestrator


def test_build_steering_variants():
    assert isinstance(build_steering({"default": "noop", "per_agent": {}}), NoOpSteering)
    pi = build_steering({"default": "prompt_injection",
                         "per_agent": {"player_0": {"system_suffix": " win."}}})
    assert isinstance(pi, PromptInjectionSteering)
    act = build_steering({"default": "activation",
                          "per_agent": {"player_0": {"layer": "l", "vector_path": "v.pt",
                                                     "coefficient": 3.0}}})
    assert isinstance(act, ActivationSteering)


def test_run_config_parses_dict():
    cfg = RunConfig.from_dict({
        "run_id": "t", "game": {"family": "symbolic", "id": "beauty_contest"},
        "episodes": 1, "agents": {"count": 2, "max_parse_retries": 3},
        "model": {"backend": "vllm", "model_id": "m"},
        "steering": {"default": "noop", "per_agent": {}},
        "logging": {"dir": "logs/"},
    })
    assert cfg.game_family == "symbolic"
    assert cfg.game_id == "beauty_contest"
    assert cfg.num_players == 2


def test_end_to_end_symbolic_with_stub(tmp_path):
    env, renderer, parser = build_game(
        family="symbolic", game_id="gbs", num_players=3,
        env_kwargs={"num_rounds": 3, "target": 50})
    policy = StubPolicy(scripted={}, default="GUESS: 50")
    logger = EpisodeLogger(run_dir=str(tmp_path), run_id="e2e", episode=0)
    orch = Orchestrator(env=env, renderer=renderer, parser=parser, policy=policy,
                        steering=NoOpSteering(), logger=logger, game="gbs",
                        max_parse_retries=2)
    final = orch.run_episode()
    # all guess 50 == target -> all rewarded, game ends round 1
    assert final == {"player_0": 1.0, "player_1": 1.0, "player_2": 1.0}
    assert (tmp_path / "e2e" / "episode_0.jsonl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_and_e2e.py -q`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write `testbed/config.py`**

```python
"""Config parsing + builders for steering and policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from testbed.steering.activation import ActivationSteering
from testbed.steering.noop import NoOpSteering
from testbed.steering.prompt_injection import PromptInjectionSteering


def build_steering(cfg: Dict[str, Any]):
    method = cfg.get("default", "noop")
    per_agent = cfg.get("per_agent", {}) or {}
    if method == "noop":
        return NoOpSteering()
    if method == "prompt_injection":
        return PromptInjectionSteering(per_agent=per_agent)
    if method == "activation":
        return ActivationSteering(per_agent=per_agent)
    raise ValueError(f"Unknown steering method: {method}")


def build_policy(model_cfg: Dict[str, Any], steering=None):
    backend = model_cfg.get("backend", "transformers")
    model_id = model_cfg["model_id"]
    if backend == "transformers":
        from testbed.policy.transformers_policy import TransformersPolicy
        return TransformersPolicy(
            model_id=model_id,
            temperature=model_cfg.get("temperature", 0.7),
            steering=steering)
    if backend == "vllm":
        from testbed.policy.vllm_policy import VLLMPolicy
        return VLLMPolicy(
            model_id=model_id,
            endpoint=model_cfg.get("endpoint", "http://localhost:8000") + "/v1",
            temperature=model_cfg.get("temperature", 0.7))
    raise ValueError(f"Unknown backend: {backend}")


@dataclass
class RunConfig:
    run_id: str
    game_family: str
    game_id: str
    episodes: int
    num_players: Optional[int]
    max_parse_retries: int
    model: Dict[str, Any]
    steering: Dict[str, Any]
    logging_dir: str
    env_kwargs: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        game = d["game"]
        agents = d.get("agents", {})
        return cls(
            run_id=d["run_id"],
            game_family=game["family"],
            game_id=game["id"],
            episodes=d.get("episodes", 1),
            num_players=agents.get("count"),
            max_parse_retries=agents.get("max_parse_retries", 5),
            model=d.get("model", {}),
            steering=d.get("steering", {"default": "noop", "per_agent": {}}),
            logging_dir=d.get("logging", {}).get("dir", "logs/"),
            env_kwargs=game.get("env_kwargs", {}),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_and_e2e.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Write `config/run_config.yaml`**

```yaml
run_id: beauty_contest_baseline_01

game:
  family: symbolic            # symbolic | textarena
  id: beauty_contest          # beauty_contest | gbs | <TextArena env_id>
  env_kwargs:
    num_rounds: 5
episodes: 1

model:
  backend: transformers       # transformers (steering) | vllm (fast baseline)
  model_id: Qwen/Qwen2.5-3B-Instruct
  endpoint: http://localhost:8000
  temperature: 0.7

agents:
  count: 4
  concurrency: sequential
  max_parse_retries: 5

steering:
  default: noop               # noop | prompt_injection | activation
  per_agent: {}

logging:
  dir: logs/
```

- [ ] **Step 6: Write `scripts/run_episode.py`**

```python
"""CLI: run episode(s) from a YAML config."""
from __future__ import annotations

import argparse
import sys

import yaml

from testbed.config import RunConfig, build_policy, build_steering
from testbed.logging_.episode_logger import EpisodeLogger
from testbed.orchestrator import Orchestrator
from testbed.registry import build_game


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = RunConfig.from_dict(raw)

    steering = build_steering(cfg.steering)
    policy = build_policy(cfg.model, steering=steering)

    for ep in range(cfg.episodes):
        env, renderer, parser_ = build_game(
            family=cfg.game_family, game_id=cfg.game_id,
            num_players=cfg.num_players or 3, env_kwargs=cfg.env_kwargs)
        logger = EpisodeLogger(run_dir=cfg.logging_dir, run_id=cfg.run_id, episode=ep)
        orch = Orchestrator(
            env=env, renderer=renderer, parser=parser_, policy=policy,
            steering=steering, logger=logger, game=cfg.game_id,
            max_parse_retries=cfg.max_parse_retries)
        final = orch.run_episode()
        print(f"Episode {ep} final rewards: {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Commit**

```bash
git add testbed/config.py config/run_config.yaml scripts/run_episode.py tests/test_config_and_e2e.py
git commit -m "feat: config loader, builders, run_episode CLI, e2e wiring"
```

---

## Task 18: Full suite green + README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest -q`
Expected: all tests pass; a few `skipped` (the `gpu`/importorskip tests). No failures, no errors.

- [ ] **Step 2: Write `README.md`** documenting setup and runs

```markdown
# LLM Steering Multi-Game Testbed

Run small local LLM agents across text-based multi-agent games to study steering
(steering vectors + prompt injection). See the design spec in
`docs/superpowers/specs/` and the plan in `docs/superpowers/plans/`.

## Games
- Symbolic (implemented directly): `beauty_contest`, `gbs`
- TextArena multiplayer (via adapter): any 3+ player TextArena `env_id`
  (e.g. `SecretMafia-v0`, `Taboo-v0`, `ThreePlayerTicTacToe-v0`, ...)

## Install
```
pip install -r requirements.txt
```

## Test
```
python -m pytest -q
```
GPU/lib-gated tests (real Qwen, live vLLM, real TextArena) are skipped unless the
hardware/libraries are present.

## Run an episode
```
python scripts/run_episode.py --config config/run_config.yaml
```
Edit `config/run_config.yaml` to pick the game, model backend (`transformers` for
steering, `vllm` for fast baselines), agent count, and steering method.

## Steering
- `noop` — baseline
- `prompt_injection` — per-agent system/user prompt edits
- `activation` — load a vector (`.pt`/`.npy`) and add it to a layer's residual
  stream via a forward hook (needs the `transformers` backend). Deriving
  meaningful vectors is done offline; the apply-path is implemented and tested.

Logs: one JSONL trace per agent per turn under `logs/<run_id>/`, plus a summary.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: project README"
```

---

## Self-Review Notes

- **Spec coverage:** EnvAdapter/turn model (T3,4,5,13), TextRenderer (T6,13),
  ActionParser w/ feedback (T7,13), Policy two backends (T10,15,16),
  SteeringMethod incl. implemented activation (T8,9,15), Orchestrator sequential
  loop + retries (T12), Logger full traces no activations (T11), registry (T14),
  config/CLI (T17), tests incl. steering apply-path on CPU (T9) and hook lifecycle
  (T15). Async concurrency intentionally out of scope (interface shaped, not built).
- **Gating:** real-model/textarena tests use `@pytest.mark.gpu` / `importorskip`
  so the suite is green on this CPU-only machine while the code is real.
- **Type consistency:** `StepResult`, `ParsedAction`/`ParseError`, `SteeringSpec`,
  `RenderContext` used consistently; `Orchestrator` passes `env.context` to
  renderer/parser (symbolic adapters expose `.context`; TextArena adapter also
  exposes `.context`).
- **Note for executor:** in Task 14 Step 3, do NOT include the intentionally-wrong
  `_unused` import; use the corrected import block shown immediately below it.
```
