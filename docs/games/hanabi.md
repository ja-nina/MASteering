# Hanabi

**Family:** `symbolic` | **ID:** `hanabi` | **Status:** skeleton — `submit()` not yet implemented

---

## Game overview

Hanabi is a cooperative card game with **imperfect self-information**: you can see
all other players' hands but not your own.  Players must use limited verbal hints
to help each other play cards in the right order onto five colour fireworks piles.

The theoretical maximum score is 25 (5 colours × ranks 1–5 each played exactly
once in order).  A misplay costs a fuse token; three misplays end the game.
A used clue token is spent until reclaimed by a discard.

This game is a strong test of **Theory-of-Mind reasoning** (inferring what others
know about your cards from the hints they give) and **cooperative communication
under a token budget**.

Reference: Bard et al. (2019). The Hanabi Challenge: A New Frontier for AI Research. AIJ.

---

## Action space (per turn, one player)

| Action | Format | Description |
|---|---|---|
| Play a card | `PLAY <pos>` | Attempt to extend a colour pile; fuse lost if wrong |
| Discard | `DISCARD <pos>` | Discard a card; gain one clue token (max 8) |
| Hint colour | `HINT player_X COLOR red` | Tell a player which of their cards are that colour |
| Hint rank | `HINT player_X RANK 3` | Tell a player which of their cards are that rank |

Positions are 1-indexed from the left of the player's hand.

---

## Observation structure

Each player sees:
- All **other** players' cards (colour + rank)
- Their own hand as **slot positions with accumulated hints** only
- The fireworks piles (current top rank per colour)
- The discard pile
- Clue tokens remaining and fuse tokens remaining

The adapter stores `clue_knowledge[agent_id][slot]` as sets of confirmed
`colors` and `ranks`, plus negative clues (`not_colors`, `not_ranks`).

---

## Config keys (`env_kwargs`)

| Key | Default | Description |
|---|---|---|
| `num_players` | 3 | 2–5 players; hand size is 5 at 2–3p, 4 at 4–5p |
| `seed` | 0 | RNG seed for deck shuffle |

---

## Implementation status

| File | Status |
|---|---|
| `testbed/envs/symbolic/hanabi.py` | Deck + hand + clue state initialised; `submit()` stubs need implementing |
| `testbed/renderers/symbolic/hanabi.py` | Complete — shows others' hands, own hint slots, fireworks |
| `testbed/parsers/symbolic/hanabi.py` | Complete — parses PLAY/DISCARD/HINT COLOR/HINT RANK |
| `testbed/registry.py` | Lazy-loaded via `_SYMBOLIC_LAZY["hanabi"]` |

**To activate:** implement `submit()` in `testbed/envs/symbolic/hanabi.py` (the docstring includes detailed pseudocode), then un-comment `"hanabi"` in `testbed/registry.py`.

---

## Key research questions

- Can LLM agents develop implicit "hint conventions" (like the Hat-Guessing strategy)?
- Does chain-of-thought / thinking mode improve Hanabi scores by enabling more careful inference?
- How does team composition of personas affect score — e.g., does a `cautious` persona play fewer risky cards?
- Does a ToM instruction improve hint interpretation?

---

## Implementation notes

- `pending()` is fully implemented — returns only the current player each step.
- `context.extra["current_player_idx"]` tracks whose turn it is.
- The renderer shows `"??"` entries for own-hand slots with no accumulated hints.
- Hint application logic must update both `colors`/`ranks` (positive) and `not_colors`/`not_ranks` (negative) for every slot in the target's hand.
- Draw logic: when the deck is empty, each player gets exactly one more turn (tracked via `turns_after_deck_empty`).

---

## Quick smoke test (after implementing submit)

```bash
python scripts/run_episode.py \
  --config config/games/hanabi/hanabi_noop_3p.yaml \
  --episodes 1
```
