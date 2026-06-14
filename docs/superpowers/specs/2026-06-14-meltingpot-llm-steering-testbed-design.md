# MeltingPot LLM Steering Testbed — Design

**Date:** 2026-06-14
**Status:** Approved design, ready for implementation planning

## 1. Purpose

Build a testbed for running small, locally-hosted LLM agents (starting with
Qwen2.5-3B-Instruct) on MeltingPot multi-agent social-dilemma games. The
**primary research goal is steering**: measuring how steering vectors and prompt
injections change agent behavior in a controlled environment. Steering vectors
themselves are derived later and outside this project's first version — the
testbed must *apply* them and leave a clean extension point, not compute them.

A hard secondary requirement is **faithful play**: agents must comprehend and
play the games competently so that behavioral changes can be attributed to
steering rather than to poor game understanding. This drives the emphasis on a
robust action-parsing layer with error feedback.

The system is designed to support **multiple agents** (up to 8, for
`rws_arena`) from the start.

## 2. Scope

### In scope (first version)
- Three MeltingPot substrates:
  - `prisoners_dilemma_in_the_matrix__repeated` (`pd`) — 2 players, mixed-motive
  - `running_with_scissors_in_the_matrix__arena` (`rws_arena`) — 8 players, zero-sum
  - `commons_harvest__open` (`harvest`) — N players, commons / resource depletion
- Language interface (TextRenderer + ActionParser) per substrate
- `Policy` abstraction wrapping a vLLM-served model
- Steering interface with no-op and prompt-injection implementations; activation
  steering present as a documented, stubbed extension point
- Sequential per-step agent execution (config-selectable; only sequential
  implemented)
- Full-trace logging (prompt + completion + action + reward per agent per step)
- Config-driven runs (substrate, model, n_agents, steering, concurrency)

### Out of scope (first version)
- Computing/deriving steering vectors (done offline, later)
- Activation-steering *implementation* inside vLLM (interface only)
- Concurrent/async agent execution (interface defined, not implemented)
- Substrates beyond the three above (architecture leaves room; not built)
- Non-MeltingPot games (Tier 1/2 from the games brief) — architecture leaves
  room via `EnvAdapter`, not built

## 3. Background & Reference

We use the **Hypothetical-Minds** repo (cloned read-only into
`reference/Hypothetical-Minds/`) as a jumping-off point. Key facts established
by reading its source:

- MeltingPot emits **pixel** observations (`obs['player_0']['WORLD.RGB']`) plus
  some structured fields (e.g. `INVENTORY`). Spatial info (positions/orientations
  of all entities) is recovered by decoding the global RGB image.
- HM decodes the image in `llm_plan/env/mp_llm_env.py`:
  `image_to_state()` cuts the frame into 8×8 sprite patches and labels each by
  **exact pixel match** against a sprite database (KNN fallback), producing
  `{entity_label: [(x,y), ...]}`. `get_ego_state()` filters to a per-agent
  egocentric window (5×5 normal, 11×11 arena). Inventory is read directly from
  the obs dict.
- HM's prompt construction and action vocabulary live in per-game agent files
  (e.g. `llm_plan/agent/prisoners_dilemma_in_the_matrix__repeated/pd_react.py`).
  Agents output a Python dict with high-level actions `move_to(src, target)` and
  `fire_at(target)`; `action_funcs.py` turns these into discrete env actions via
  A* pathfinding. Invalid plans are re-prompted with templated error feedback
  (`is_valid_plan`, the `subgoal_module` retry loop).
- HM already runs open models through **vLLM's OpenAI-compatible HTTP server**
  (README lines 48–65). That server is great for prompt-injection but hides
  model internals, so **activation steering needs custom vLLM forward hooks**
  rather than plain HTTP requests.

The "games brief" (`games_brief.txt`) frames the correct abstraction: the spine
of an LLM game testbed is the **language interface** — a `TextRenderer`
(state → prompt) and an `ActionParser` (LLM text → valid action, with error
feedback) — with the model behind a `Policy.act()` and the environment beneath
an adapter. MeltingPot games are "Tier 3" (pixel → need a real text-abstraction
layer = the HM renderer). Our architecture makes these abstractions first-class
so Tier 1/2 games can slot in later.

## 4. Architecture Overview

```
                   ┌─────────────────────────────────────────┐
                   │              Orchestrator                  │
                   │   (game loop: render → act → parse → step) │
                   └─────────────────────────────────────────┘
                          │            │             │
        ┌─────────────────┘            │             └──────────────────┐
        ▼                              ▼                                 ▼
┌───────────────┐          ┌─────────────────────┐            ┌──────────────────┐
│ EnvAdapter    │          │  Policy.act(prompt)  │            │  Logger          │
│ (MeltingPot)  │          │   vLLM + steering    │            │ (jsonl traces)   │
└───────────────┘          └─────────────────────┘            └──────────────────┘
        │ raw obs                  ▲          │ raw text
        ▼                         │           ▼
┌───────────────┐    prompt      │     ┌──────────────────┐   action idx
│ TextRenderer  │────────────────┘     │  ActionParser     │──────────────►
│ (per game)    │                      │  (per game,       │   to EnvAdapter
│ state → text  │                      │   error feedback) │
└───────────────┘                      └──────────────────┘
```

The Orchestrator is the only component that knows about all the others. Every
other module communicates through a narrow interface and can be understood,
tested, and replaced independently.

## 5. Components & Interfaces

### 5.1 EnvAdapter
Wraps a MeltingPot substrate; the only component touching MeltingPot APIs.

```python
class EnvAdapter(Protocol):
    def reset(self) -> tuple[dict, dict]:      # (raw_obs_per_agent, info)
    def step(self, actions: dict[str, int]) -> tuple[dict, dict, bool, dict]:
        # (raw_obs_per_agent, rewards_per_agent, done, info)
    def agent_ids(self) -> list[str]
    def action_space_size(self, agent_id: str) -> int
    def close(self) -> None
```

Implementation reuses HM's substrate-building approach
(`meltingpot.substrate.build` + the RLLib-style multi-agent wrapper). Raw obs
include `WORLD.RGB` and per-agent fields like `INVENTORY`.

### 5.2 TextRenderer (per game) — language-out
Converts raw observation into the prompt text shown to a given agent.

```python
class TextRenderer(Protocol):
    def render(self, raw_obs: dict, agent_id: str, context: RenderContext) -> str
    def system_prompt(self, agent_id: str) -> str
```

- Internally uses the HM pixel-decode pipeline (`image_to_state`,
  `build_grid_from_states`, `get_ego_state`) adapted as our own code under
  `envs/meltingpot/state_decode.py`, plus per-game prompt templates adapted from
  HM's agent files.
- `RenderContext` carries memory of previously-seen entities, interaction
  history, last rewards, and prior execution outcomes — the fields HM threads
  through `generate_feedback_user_message`.
- One renderer per substrate: `PDRenderer`, `RWSArenaRenderer`,
  `CommonsHarvestRenderer`.
- The sprite-label databases from `reference/Hypothetical-Minds/llm_plan/sprite_labels/<substrate>/`
  are vendored into our project per substrate.

### 5.3 Policy — the model + steering
Game-agnostic. Wraps the vLLM-served model and applies steering.

```python
class Policy(Protocol):
    def act(self, system_prompt: str, user_prompt: str,
            agent_id: str, steering: SteeringSpec | None) -> str   # raw completion
```

- First implementation: `VLLMPolicy`, talking to a local vLLM server running
  `Qwen2.5-3B-Instruct` (model id from config).
- **Serving decision:** we run vLLM with **custom forward hooks** on the
  transformer layers so steering vectors can be injected (exposed as a custom
  request parameter / per-agent registration), while keeping vLLM throughput and
  an OpenAI-compatible request surface for the prompt-injection path. The hook
  machinery is built but the activation-vector application itself is a stub in
  v1 (see 5.4).

### 5.4 SteeringMethod — interface + stubs
```python
class SteeringMethod(Protocol):
    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> tuple[str, str]
    def steering_spec(self, agent_id: str) -> SteeringSpec | None
```

- `NoOpSteering` — identity; the baseline for faithful-play validation.
- `PromptInjectionSteering` — mutates system/user prompt per agent.
- `ActivationSteering` — **stub** in v1. Interface and the vLLM hook
  registration path exist; `steering_spec` returns the layer + vector to add.
  Loading precomputed vectors (`.pt`/`.npy`) and the actual hook math are
  documented extension points to fill once vectors are derived offline.
- Steering is **per-agent**: each agent slot can carry a different
  `SteeringSpec` (or none), enabling steered-vs-unsteered comparisons within one
  episode.

### 5.5 ActionParser (per game) — language-in
The faithful-play workhorse. Turns LLM text into a valid discrete action, with
error feedback for self-correction.

```python
class ActionParser(Protocol):
    def parse(self, completion: str, raw_obs: dict, agent_id: str,
              context: ParseContext) -> ParseResult
        # ParseResult = Action(int) | ActionPlan(list[int]) | ParseError(feedback_text)
```

- Extracts the Python action dict (HM's `extract_dict` / `ast.literal_eval`
  approach), validates against walls / bounds / legal actions
  (`is_valid_plan`), and translates `move_to`/`fire_at` into discrete actions
  via A* pathfinding (HM's `action_funcs.py`, adapted into
  `envs/meltingpot/action_funcs.py`).
- On invalid/unparseable output returns a **`ParseError` with templated
  feedback**; the Orchestrator re-prompts the agent (bounded retries, HM uses
  ≤10). This is the mechanism that keeps play faithful instead of silently
  NOOPing.
- Because HM agents emit multi-step plans, the parser may return a short action
  queue; the Orchestrator drains one discrete action per env step (with a
  one-step lookahead re-plan check, as in HM's `check_plan_one_step`).
- One parser per substrate.

### 5.6 Orchestrator — the game loop
Owns the episode loop and wiring. Per step, **sequentially** for each agent:

1. `TextRenderer.render` → prompt
2. `SteeringMethod.apply_to_prompt` → possibly-modified prompt
3. `Policy.act` → completion (with the agent's `SteeringSpec`)
4. `ActionParser.parse` → action or `ParseError`
5. on `ParseError`: re-prompt up to N times, else fall back to NOOP (logged)
6. collect action; `Logger.log_step`

After all agents have an action: `EnvAdapter.step(actions)`, distribute rewards,
update each renderer's `RenderContext`/memory, advance.

**Concurrency is config-selectable** (`sequential` | `async`). Only
`sequential` is implemented in v1; the agent-iteration interface is shaped so an
`async` executor (HM-style `asyncio.gather` against vLLM) can be dropped in
later without restructuring the loop.

### 5.7 Logger
Writes one JSONL line per agent per step:
`{episode, step, agent_id, system_prompt, user_prompt, completion, parsed_action,
parse_retries, reward, steering_spec_id}`. One file per episode under
`logs/<run_id>/episode_<n>.jsonl`. Per-episode summary metrics (total reward per
agent, interaction counts, parse-error rate) written to a sidecar JSON. **No
activation dumps** (deliberately, for storage cost).

## 6. Project Structure

```
MA_Environments/
├── reference/Hypothetical-Minds/      # cloned, read-only study reference (gitignored)
├── games_brief.txt                    # framing notes (language-interface spine)
├── config/
│   └── run_config.yaml                # substrate, model, n_agents, steering, concurrency
├── testbed/
│   ├── orchestrator.py                # game loop
│   ├── envs/
│   │   ├── adapter.py                 # EnvAdapter protocol
│   │   └── meltingpot/
│   │       ├── mp_adapter.py          # MeltingPot substrate wrapper
│   │       ├── state_decode.py        # adapted image_to_state / ego_state / grid
│   │       ├── action_funcs.py        # adapted A* move_to / fire_at
│   │       └── sprite_labels/<substrate>/   # vendored sprite databases
│   ├── renderers/
│   │   ├── base.py                    # TextRenderer protocol + RenderContext
│   │   ├── pd.py
│   │   ├── rws_arena.py
│   │   └── commons_harvest.py
│   ├── parsers/
│   │   ├── base.py                    # ActionParser protocol + ParseResult/ParseError
│   │   ├── pd.py
│   │   ├── rws_arena.py
│   │   └── commons_harvest.py
│   ├── policy/
│   │   ├── base.py                    # Policy protocol
│   │   └── vllm_policy.py             # vLLM client + steering-hook registration
│   ├── steering/
│   │   ├── base.py                    # SteeringMethod protocol + SteeringSpec
│   │   ├── noop.py
│   │   ├── prompt_injection.py
│   │   └── activation.py              # stub + documented extension point
│   └── logging/
│       └── episode_logger.py
├── scripts/
│   ├── serve_vllm.py / .md            # how to launch vLLM (Qwen2.5-3B) with hooks
│   └── run_episode.py                 # CLI entry point
└── docs/superpowers/specs/            # this spec
```

## 7. Data Flow (one step, one agent)

```
EnvAdapter.step() ─ raw_obs (WORLD.RGB + INVENTORY) ─► TextRenderer
  TextRenderer: decode pixels → entities → ego window → prompt text
  → SteeringMethod.apply_to_prompt → (sys, user)
  → Policy.act(sys, user, steering_spec) → completion text
  → ActionParser.parse: extract dict → validate → A* → discrete action(s)
       └─ on error: ParseError(feedback) → Orchestrator re-prompts (≤N)
  → Logger.log_step(...)
Orchestrator collects all agents' actions → EnvAdapter.step(actions)
  → rewards distributed, RenderContext/memory updated
```

## 8. Error Handling

- **Unparseable / illegal LLM output:** `ActionParser` returns templated
  feedback; Orchestrator re-prompts up to N times (config, default ~5). Exhausted
  → NOOP, flagged in logs (`parse_retries`, fallback marker). Parse-error rate
  is a tracked metric — a spike signals unfaithful play to investigate.
- **No path found (A*):** progressive relaxation of obstacles (HM pattern:
  drop same-type resources, then opponents, then everything but walls).
- **Agent removed mid-episode** (e.g. zapped out): adapter reports missing
  entity; renderer emits a "not currently in play" state; agent NOOPs until
  respawn.
- **vLLM unreachable / timeout:** Policy raises; Orchestrator aborts the run
  with a clear message rather than silently degrading (faithful-play integrity).

## 9. Testing Strategy

- **Unit — ActionParser** (highest value): feed canned completions (valid plan,
  malformed dict, wall-targeting move, out-of-bounds, empty) and assert correct
  action or correct error feedback. No model or env needed.
- **Unit — TextRenderer:** feed a saved raw_obs fixture (captured from a real
  reset) and assert the prompt contains expected positions/inventory/egocentric
  entities.
- **Unit — state_decode:** known frame → expected entity dict (exact-pixel-match
  determinism makes this stable).
- **Integration — env smoke test:** reset + random-action steps for each
  substrate, assert shapes/agent counts (`pd`=2, `rws_arena`=8, harvest=N).
- **Integration — full loop with a stub Policy** (returns canned valid plans):
  run a few steps end-to-end, assert logs written, rewards flow, no crashes.
  This validates the harness without a GPU.
- **Manual/GPU smoke:** one short episode per substrate with Qwen2.5-3B to
  confirm faithful play before steering work begins.

## 10. Configuration (example)

```yaml
run_id: pd_baseline_01
substrate: pd                 # pd | rws_arena | harvest
episodes: 1
max_steps: 1000

model:
  backend: vllm
  model_id: Qwen/Qwen2.5-3B-Instruct
  endpoint: http://localhost:8000
  temperature: 0.7

agents:
  count: null                 # null = substrate default (pd=2, rws_arena=8)
  concurrency: sequential     # sequential | async  (only sequential implemented)
  max_parse_retries: 5

steering:
  default: noop               # noop | prompt_injection | activation
  per_agent:                  # optional overrides for steered-vs-unsteered runs
    player_0: noop
    # player_1: { method: activation, layer: 14, vector: vectors/coop.pt }  # later

logging:
  dir: logs/
  full_traces: true
  activations: false
```

## 11. Key Decisions (resolved)

| Decision | Choice | Rationale |
|---|---|---|
| Primary goal | Steering research | Faithful play is the means; behavioral attribution is the end |
| State representation | Structured text via HM pixel-decode | MeltingPot is Tier-3 pixel; reuse proven HM renderer |
| Serving | vLLM + custom forward hooks | Fast + OpenAI-compatible for prompt injection; hooks enable activation steering |
| Steering in v1 | Interface + noop/prompt-injection; activation stubbed | Vectors derived later, offline |
| First model | Qwen2.5-3B-Instruct (config-driven) | Small/fast for harness iteration; swappable |
| Concurrency | Config option; sequential implemented | Debuggable first; async path left open for 8-agent throughput |
| Logging | Full traces, no activations | Auditability without prohibitive storage |
| Multi-agent | First-class, per-agent steering | rws_arena needs 8; steered-vs-unsteered within an episode |
| Extensibility | Language-interface spine (Renderer/Parser/Policy/Adapter) | Tier-1/2 non-MeltingPot games can slot in later |

## 12. Open Questions / Future Work

- Activation-steering implementation: exact vLLM hook layer API, vector format,
  and per-token vs per-sequence application — deferred until vectors exist.
- Async concurrency executor for 8-agent throughput.
- Derived per-episode behavioral metrics (cooperation rate, defection counts)
  beyond raw reward — easy to add on top of full traces.
- Additional substrates / Tier-1/2 games via new adapters + renderers + parsers.
