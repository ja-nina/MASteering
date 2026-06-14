# LLM Steering Multi-Game Testbed — Design

**Date:** 2026-06-14
**Status:** Approved design, ready for implementation planning

## 1. Purpose

Build a testbed for running small, locally-hosted LLM agents (starting with
Qwen2.5-3B-Instruct) as **multiple agents** across a range of text-based
multi-agent games. The **primary research goal is steering**: measuring how
steering vectors and prompt injections change agent behavior. Steering must work
**identically across every game** so effects can be compared across very
different strategic settings — pure numeric games and turn-based
negotiation/deduction/economic games.

The steering *application* mechanism (prompt injection and activation-vector
addition via forward hooks) is fully implemented and tested in this version. Only
the offline *derivation* of meaningful vectors is out of scope and done later; the
apply-path is verified now with synthetic vectors.

A hard secondary requirement is **faithful play**: agents must comprehend and
play the games competently so behavioral changes can be attributed to steering
rather than poor game understanding. This drives the emphasis on robust action
parsing with error feedback.

## 2. Scope

### In scope (first version) — two game families

**Family A — Native symbolic games (implemented directly, tiny state)**
- `beauty_contest` — Keynesian beauty contest (guess 2/3 of the group average),
  N players
- `gbs` — group binary search, N players
- Turn model: **simultaneous** (all players submit each round)

**Family B — TextArena multiplayer (text-native, via one adapter)**
- All 11 of TextArena's 3+ player games, config-selectable by `env_id`:
  BlindAuction, CharacterConclave, Codenames, Diplomacy, Negotiation,
  SecretMafia, Taboo, ThreePlayerGOPS, ThreePlayerIPD, ThreePlayerTicTacToe,
  TwoRoomsAndABoom
- Turn model: **turn-based** (env names the player whose turn it is)

### Cross-cutting (both families)
- `Policy` abstraction with two backends: in-process HuggingFace transformers
  (steering-capable) and a vLLM client (fast, no-steering baselines)
- Steering interface fully implemented: no-op, prompt-injection, **and
  activation steering** (load a vector, add it to the residual stream at a chosen
  layer with a coefficient via forward hooks) — applied uniformly regardless of
  game. Only the offline *derivation* of meaningful vectors is out of scope.
- Sequential per-step agent execution (config-selectable; only sequential
  implemented)
- Full-trace logging (prompt + completion + action + reward per agent per turn)
- Config-driven runs

### Out of scope (first version)
- **MeltingPot / any pixel-based games** (removed from scope)
- Computing/deriving *meaningful* steering vectors (done offline, later); the
  application mechanism IS implemented and tested with synthetic vectors
- Concurrent/async agent execution (interface defined, not implemented)
- TextArena 1- and 2-player games (only 3+ player included; trivially added later)
- Additional native symbolic games beyond beauty contest and GBS

## 3. Background & Reference

### 3.1 The language-interface spine
The "games brief" (`games_brief.txt`) frames the abstraction: the spine of an LLM
game testbed is the **language interface** — a `TextRenderer` (state → prompt) and
an `ActionParser` (LLM text → valid action, with error feedback) — with the model
behind a `Policy.act()` and the environment beneath an `EnvAdapter`. Games fall
into tiers by how much language-interface work they need:

- **Tier 1 (TextArena):** already text-native; the framework supplies both halves.
- **Tier 2 (symbolic):** trivial to write; state is a number or a short summary.

(Tier 3 = pixel games such as MeltingPot — explicitly out of scope.)

Our architecture makes these abstractions first-class so both families share one
orchestrator, one policy, one steering layer, and one logger.

### 3.2 TextArena (Tier-1 reference)
Gym-style, **turn-based** API:
```python
env = ta.make(env_id="SecretMafia-v0"); env.reset(num_players=N)
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)   # string in, string out
    done, info = env.step(action=action)
rewards, game_info = env.close()
```
The env validates actions and produces observation strings itself, so our adapter
relays text rather than implementing per-game renderers/parsers.

## 4. Architecture Overview

```
                   ┌─────────────────────────────────────────────┐
                   │                Orchestrator                    │
                   │  loop: pending() → render → act → parse →      │
                   │        submit()  (uniform across game families)│
                   └─────────────────────────────────────────────┘
                          │            │             │
        ┌─────────────────┘            │             └──────────────────┐
        ▼                              ▼                                 ▼
┌───────────────┐          ┌─────────────────────┐            ┌──────────────────┐
│ EnvAdapter    │          │  Policy.act(prompt)  │            │  Logger          │
│  Symbolic   │ │          │   vLLM + steering    │            │ (jsonl traces)   │
│  TextArena  │ │          └─────────────────────┘            └──────────────────┘
└───────────────┘                  ▲          │
        │ pending()=(agent,obs)    │ prompt   │ completion
        ▼                          │          ▼
┌───────────────┐    prompt       │     ┌──────────────────┐   action
│ TextRenderer  │─────────────────┘     │  ActionParser     │──────────────►
│ (per game;    │                       │  (per game;       │  to EnvAdapter
│  pass-through │                       │   pass-through for │   .submit()
│  for TextArena)│                      │   TextArena)      │
└───────────────┘                       └──────────────────┘
```

The Orchestrator is the only component aware of all others, and it is written
**once** against the generalized adapter interface — it does not branch on game
family. Turn-based vs simultaneous is absorbed entirely by `pending()`/`submit()`.

## 5. Components & Interfaces

### 5.1 EnvAdapter (generalized turn model)
The single abstraction that unifies simultaneous and turn-based games.

```python
class EnvAdapter(Protocol):
    def reset(self) -> None
    def pending(self) -> list[tuple[str, RawObs]]
        # agents that must act now: ALL for simultaneous, ONE for turn-based
    def submit(self, actions: dict[str, Action]) -> StepResult
        # StepResult = (rewards: dict, done: bool, info: dict); advances the env
    def agent_ids(self) -> list[str]
    def legal_actions(self, agent_id: str) -> ActionSpace | None
    def close(self) -> dict           # final rewards / game info
```

Two implementations:
- **`SymbolicAdapter`** (base for beauty_contest, gbs) — `pending()` returns all
  players; `submit()` runs the (tiny) game rule and returns rewards. Rules
  implemented directly.
- **`TextArenaAdapter`** — wraps `ta.make`; `pending()` returns the single player
  from `get_observation()`; `submit({player: action})` calls `env.step`; `close()`
  returns TextArena rewards. One adapter serves all 11 multiplayer games via the
  configured `env_id`.

### 5.2 TextRenderer (per game) — language-out
```python
class TextRenderer(Protocol):
    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str
    def system_prompt(self, agent_id: str) -> str
```
- **Symbolic:** a few f-strings (e.g. "Round 3. Last round's group average was
  41.2; the target (2/3 of average) was 27.5. Choose an integer 0–100.").
  `BeautyContestRenderer`, `GBSRenderer`.
- **TextArena:** `TextArenaRenderer` is **pass-through** — relays the observation
  string TextArena already produced, optionally prepending a system prompt. No
  per-game work.

### 5.3 ActionParser (per game) — language-in (faithful-play workhorse)
```python
class ActionParser(Protocol):
    def parse(self, completion: str, raw_obs: RawObs, agent_id: str,
              context: ParseContext) -> ParseResult
        # ParseResult = Action | ParseError(feedback_text)
```
- **Symbolic:** parse a number/choice from text; range-check. Tiny.
- **TextArena:** `TextArenaParser` is **near pass-through** — TextArena validates
  actions itself, so the parser extracts the action string (e.g. the bracketed
  token games expect) from the completion and forwards it; TextArena's own error
  handling drives any feedback.
- On invalid output, parser returns **`ParseError` with templated feedback**; the
  Orchestrator re-prompts (bounded retries, default ~5). This keeps play faithful
  instead of silently defaulting.

### 5.4 Policy — the model + steering (game-agnostic)
```python
class Policy(Protocol):
    def act(self, system_prompt: str, user_prompt: str,
            agent_id: str, steering: SteeringSpec | None) -> str   # raw completion
```
Two interchangeable backends, selected by config:

- **`TransformersPolicy`** (steering-capable, default for steering runs) — loads
  `Qwen2.5-3B-Instruct` in-process via HuggingFace `transformers`. Registers
  PyTorch **forward hooks** on the decoder layer named in a `SteeringSpec` and
  adds `coefficient * vector` to that layer's residual-stream output during
  generation. This is the standard, robust way to do activation steering (cf.
  `repeng` / `steering-vectors`). Slower than vLLM but correct and easy to verify;
  fine for small models and sequential play.
- **`VLLMPolicy`** (fast, no-steering baselines) — OpenAI-compatible client to a
  local vLLM server. Used for prompt-injection and no-op runs where activation
  steering is not needed and throughput matters.

The two backends are behind the same `Policy.act` interface; `agents.backend`
in config picks one. Activation steering requires `TransformersPolicy`; selecting
`VLLMPolicy` with an `ActivationSteering` spec is a config error caught at startup.

### 5.5 SteeringMethod — interface + stubs (uniform across all games)
```python
class SteeringMethod(Protocol):
    def apply_to_prompt(self, system_prompt: str, user_prompt: str,
                        agent_id: str) -> tuple[str, str]
    def steering_spec(self, agent_id: str) -> SteeringSpec | None
```
- `NoOpSteering` — identity; baseline for faithful-play validation.
- `PromptInjectionSteering` — mutates system/user prompt per agent.
- `ActivationSteering` — **fully implemented**. A `SteeringSpec` carries
  `{layer, vector_path, coefficient}`. The method loads the vector (`.pt`/`.npy`),
  and `TransformersPolicy` registers a forward hook on the named layer that adds
  `coefficient * vector` to the residual stream for that agent's generation, then
  removes the hook afterward (so per-agent specs don't leak across agents). The
  only deferred piece is *obtaining meaningful* vectors (offline contrastive
  derivation); the apply-path is real and verified with synthetic vectors (see
  Testing).
- Steering is **per-agent** and **game-agnostic**: any agent in any game can carry
  a `SteeringSpec`, enabling steered-vs-unsteered comparisons within one game and
  across games.

### 5.6 Orchestrator — the (single) game loop
Written once against `EnvAdapter`; does not branch on game family. Each iteration:

1. `agents_now = adapter.pending()`  (all, or one for turn-based)
2. for each `(agent_id, raw_obs)` in `agents_now` (**sequentially** in v1):
   a. `TextRenderer.render` → prompt
   b. `SteeringMethod.apply_to_prompt` → possibly-modified prompt
   c. `Policy.act` → completion (with agent's `SteeringSpec`)
   d. `ActionParser.parse` → action or `ParseError`
   e. on `ParseError`: re-prompt up to N; else safe default, flagged in logs
   f. `Logger.log_step(...)`
3. `adapter.submit(actions)` → rewards/done; update `RenderContext`/memory
4. repeat until done; `adapter.close()` → final rewards

**Concurrency** is config-selectable (`sequential` | `async`); only `sequential`
is implemented. The agent-iteration step is shaped so an `async` executor
(`asyncio.gather` against vLLM) can replace the per-agent loop in step 2 for
simultaneous games without restructuring.

### 5.7 Logger
One JSONL line per agent per turn:
`{game, episode, turn, agent_id, system_prompt, user_prompt, completion,
parsed_action, parse_retries, reward, steering_spec_id}`. One file per episode
under `logs/<run_id>/episode_<n>.jsonl`. Per-episode summary metrics (total reward
per agent, parse-error rate, game-specific counters) in a sidecar JSON. **No
activation dumps** (storage cost).

## 6. Project Structure

```
MA_Environments/
├── games_brief.txt                    # framing notes (language-interface spine)
├── config/
│   └── run_config.yaml
├── testbed/
│   ├── orchestrator.py                # single game loop (pending/submit)
│   ├── envs/
│   │   ├── adapter.py                 # EnvAdapter protocol + RawObs/StepResult
│   │   ├── symbolic/
│   │   │   ├── base.py                # SymbolicAdapter base
│   │   │   ├── beauty_contest.py
│   │   │   └── gbs.py
│   │   └── textarena/
│   │       └── ta_adapter.py          # wraps ta.make; all 11 multiplayer games
│   ├── renderers/
│   │   ├── base.py                    # TextRenderer + RenderContext
│   │   ├── symbolic/ (beauty_contest.py, gbs.py)
│   │   └── textarena.py               # pass-through renderer
│   ├── parsers/
│   │   ├── base.py                    # ActionParser + ParseResult/ParseError
│   │   ├── symbolic/ (beauty_contest.py, gbs.py)
│   │   └── textarena.py               # near pass-through parser
│   ├── policy/
│   │   ├── base.py                    # Policy protocol
│   │   ├── transformers_policy.py     # in-process HF model + steering hook integration
│   │   └── vllm_policy.py             # fast OpenAI-compatible client (no steering)
│   ├── steering/
│   │   ├── base.py                    # SteeringMethod + SteeringSpec
│   │   ├── noop.py
│   │   ├── prompt_injection.py
│   │   ├── activation.py              # implemented: vector load + hook fn factory
│   │   └── vectors/                   # precomputed vectors dropped here (gitignored)
│   ├── registry.py                    # game_id → (adapter, renderer, parser)
│   └── logging/
│       └── episode_logger.py
├── scripts/
│   ├── serve_vllm.py / .md            # launch vLLM (Qwen2.5-3B) with hooks
│   └── run_episode.py                 # CLI entry point
└── docs/superpowers/specs/            # this spec
```

A small `registry.py` maps a `game_id` to its `(adapter, renderer, parser)`
triple, so adding a game is a registry entry plus its classes — the
orchestrator/policy/steering/logger never change.

## 7. Data Flow (one agent acting)

```
adapter.pending() ─ (agent_id, raw_obs) ─► TextRenderer.render → prompt
  → SteeringMethod.apply_to_prompt → (sys, user)
  → Policy.act(sys, user, steering_spec) → completion text
  → ActionParser.parse → action | ParseError(feedback) → re-prompt (≤N)
  → Logger.log_step(...)
(after all pending agents acted) adapter.submit(actions)
  → rewards/done, RenderContext/memory updated
```

Simultaneous games (symbolic) yield all agents from `pending()` per loop;
turn-based games (TextArena) yield one. The flow above is identical either way.

## 8. Error Handling

- **Unparseable/illegal LLM output:** `ActionParser` returns templated feedback;
  Orchestrator re-prompts up to N (default ~5). Exhausted → safe default, flagged.
  Parse-error rate is a tracked metric (spike ⇒ unfaithful play to investigate).
- **Agent removed mid-episode** (e.g. eliminated in SecretMafia): adapter omits it
  from `pending()`.
- **TextArena invalid action:** TextArena handles it internally (may penalize/skip);
  we log its `step_info`.
- **vLLM unreachable/timeout:** Policy raises; Orchestrator aborts the run with a
  clear message rather than silently degrading (faithful-play integrity).

## 9. Testing Strategy

- **Unit — ActionParser (highest value):** canned completions per family (valid,
  malformed, out-of-range/illegal) → assert action or correct error feedback. No
  model/env needed.
- **Unit — TextRenderer:** fixtures → assert prompt content (symbolic round
  summary; TextArena pass-through relays observation verbatim).
- **Integration — adapters:** for each family, reset + a few `pending()/submit()`
  cycles with valid actions; assert turn model (symbolic=all, TextArena=one),
  agent counts, reward shapes.
- **Integration — full loop with stub Policy** (returns canned valid actions per
  family): run a few turns end-to-end per family; assert logs written, rewards
  flow, no crashes. Validates the harness with no GPU.
- **Steering — apply-path verification (GPU):** with `TransformersPolicy` and a
  **synthetic** vector, assert (a) the forward hook fires at the configured layer,
  (b) hidden states / logits differ from the no-op run, (c) a large coefficient
  measurably shifts outputs, and (d) the hook is removed after generation so an
  unsteered agent in the same episode is unaffected. This proves the mechanism
  before any meaningful vector exists.
- **Manual/GPU smoke:** one short episode per game with Qwen2.5-3B to confirm
  faithful play before steering experiments begin.

## 10. Configuration (example)

```yaml
run_id: beauty_contest_baseline_01

game:
  family: symbolic            # symbolic | textarena
  id: beauty_contest          # beauty_contest|gbs | TextArena env_id
  # family: textarena
  # id: SecretMafia-v0        # example TextArena multiplayer game
episodes: 1
max_steps: 1000

model:
  backend: transformers       # transformers (steering-capable) | vllm (fast, no steering)
  model_id: Qwen/Qwen2.5-3B-Instruct
  endpoint: http://localhost:8000   # used only by vllm backend
  temperature: 0.7

agents:
  count: 5                    # for symbolic games; TextArena uses env default unless overridable
  concurrency: sequential     # sequential | async  (only sequential implemented)
  max_parse_retries: 5

steering:
  default: noop               # noop | prompt_injection | activation
  per_agent:
    player_0: noop
    player_1:                 # activation steering is implemented; needs transformers backend
      method: activation
      layer: model.layers.14
      vector: testbed/steering/vectors/coop.pt
      coefficient: 8.0

logging:
  dir: logs/
  full_traces: true
  activations: false
```

## 11. Key Decisions (resolved)

| Decision | Choice | Rationale |
|---|---|---|
| Primary goal | Steering research, applied uniformly across games | Compare a vector's effect across numeric and social games |
| Game families in v1 | Symbolic (beauty_contest, gbs) + all TextArena 3+ player games | Text-native foundation; MeltingPot dropped |
| MeltingPot / pixel games | **Out of scope** | Removed at user request; text-native games are a cleaner base for steering |
| Turn model | Generalized `pending()`/`submit()` adapter | One orchestrator serves simultaneous and turn-based games |
| Symbolic games | Implemented directly | State is tiny; renderer/parser are a few lines |
| TextArena games | Single adapter, all multiplayer games via `env_id` | Text-native; framework supplies the language interface |
| Serving | Two backends: transformers in-process (steering) + vLLM client (fast baselines) | Robust activation steering via HF forward hooks; vLLM kept for throughput on no-steering runs |
| Steering in v1 | noop + prompt-injection + **activation, all implemented** | Apply-path is real and tested with synthetic vectors; only meaningful-vector derivation is deferred |
| First model | Qwen2.5-3B-Instruct (config-driven) | Small/fast for iteration; swappable |
| Concurrency | Config option; sequential implemented | Debuggable first; async path left open |
| Logging | Full traces, no activations | Auditability without prohibitive storage |
| Extensibility | Registry: game_id → (adapter, renderer, parser) | Adding a game never touches the core spine |

## 12. Open Questions / Future Work

- Offline derivation of *meaningful* steering vectors (contrastive activation
  pairs, etc.) — the only deferred part of the steering work; the apply-path is
  implemented now.
- Steering application detail to settle during implementation: per-token vs
  per-sequence addition, and which layer(s) work best for Qwen2.5-3B — the
  interface supports a configurable layer + coefficient so this is a tuning, not
  redesign, question.
- Async concurrency executor for high-agent-count simultaneous games.
- Derived per-game behavioral metrics (cooperation rate, beauty-contest
  convergence toward 0) beyond raw reward — easy to add on full traces.
- TextArena 1-/2-player games and additional symbolic games — each a registry
  entry plus (mostly trivial) classes.
- Turn-based steering nuance: for long TextArena dialogues, whether to steer every
  turn or only decision turns.
- Whether TextArena's `num_players` is freely configurable per game or fixed by
  `env_id` (affects agent-count config) — verify during implementation.
