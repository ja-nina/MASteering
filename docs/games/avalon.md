# The Resistance: Avalon

**Family:** `symbolic` | **ID:** `avalon` | **Status:** skeleton — `submit()` not yet implemented

---

## Game overview

5–10 players are secretly divided into two sides.  **Good** players (Merlin,
Percival, Loyal Servants) must successfully complete 3 of 5 quests.  **Evil**
players (Mordred, Morgana, Spies) must cause 3 quest failures — or, if Good wins
3 quests, correctly assassinate Merlin to steal the win.

This is a game of hidden information, deduction, and deliberate misdirection.
LLM agents must reason about who is likely evil (or pretend to be good) using
only the public vote history, quest results, and the statements other players make.

---

## Phase structure

Each round cycles through these phases:

| Phase | `pending()` returns | Action format |
|---|---|---|
| `propose` | leader only | `TEAM: player_0, player_2` |
| `team_vote` | all living | `VOTE: APPROVE` or `VOTE: REJECT` |
| `quest_vote` | team members only | `QUEST: SUCCESS` or `QUEST: FAIL` |
| `assassinate` | first spy (assassin) | `ASSASSINATE: player_X` |

Five consecutive rejected proposals → Evil wins automatically.

---

## Role knowledge

| Role | What they see |
|---|---|
| Merlin | All evil players except Mordred |
| Percival | Two players — one is Merlin, one may be Morgana (indistinguishable) |
| Loyal Servant | Nothing |
| Spy / Mordred | Each other (except Oberon, who is isolated) |
| Morgana | Other spies (appears as Merlin to Percival) |
| Oberon | Nothing — isolated evil, unknown to other spies |

---

## Config keys (`env_kwargs`)

| Key | Default | Description |
|---|---|---|
| `num_players` | 5 | 5–10 players |
| `include_merlin` | `true` | Add Merlin and the assassination phase |
| `include_percival` | `false` | Add Percival + Morgana roles |
| `include_mordred` | `false` | Add Mordred (hidden from Merlin) |
| `include_morgana` | `false` | Add Morgana (appears as Merlin) |
| `include_oberon` | `false` | Add Oberon (isolated evil) |
| `seed` | 0 | RNG seed for role assignment |

---

## Implementation status

| File | Status |
|---|---|
| `testbed/envs/symbolic/avalon.py` | State + phase management done; `submit()` stubs need implementing |
| `testbed/renderers/symbolic/avalon.py` | Complete — system prompt + per-phase render |
| `testbed/parsers/symbolic/avalon.py` | Complete — parses all 4 phase formats |
| `testbed/registry.py` | Lazy-loaded via `_SYMBOLIC_LAZY["avalon"]` |

**To activate:** implement the 4 `raise NotImplementedError` branches in `submit()`, then un-comment the `"avalon"` entry in `_SYMBOLIC` in `testbed/registry.py`.

---

## Key research questions

- Does Merlin successfully hide from the Spy team across extended play?
- How does ToM steering change bluffing strategy vs. truth-telling?
- Do personas (e.g., `risk_averse`, `contrarian`) affect quest-vote behaviour in distinguishable ways?
- Can LLMs sustain consistent hidden-role deception across 5+ rounds of discussion?

---

## Implementation notes

- `pending()` is fully implemented — it returns the correct agent subset per phase.
- `context.extra["seer_knowledge"]` is used here for Merlin's knowledge (rename if confusing).
- The assassination phase only fires when Good wins 3 quests AND `include_merlin=True`.
- Quest team size and fail requirements follow standard Avalon rules (see `_QUEST_TEAM_SIZES` in adapter).

---

## Quick smoke test (after implementing submit)

```bash
python scripts/run_episode.py \
  --config config/games/avalon/avalon_noop_5p.yaml \
  --episodes 1
```
