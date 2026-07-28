# Overcooked (text-based)

**Family:** `symbolic` | **ID:** `overcooked` | **Status:** skeleton — grid simulation not yet implemented

---

## Game overview

A cooperative, time-pressured kitchen coordination game adapted for LLM agents.
Two players share a grid kitchen: they must pick up ingredients, add them to pots,
wait for them to cook, and deliver finished dishes to the serving window — all
while coordinating their movements through a tight space.

This game tests **real-time cooperative planning and implicit coordination**:
each player must reason about what their partner is doing and select
complementary actions without explicit communication.

Reference game: Carroll et al. (2019). On the Utility of Learning about Humans
for Human-AI Coordination. NeurIPS 2019.

---

## Action space (simultaneous, every step)

| Action | Description |
|---|---|
| `MOVE N/S/E/W` | Move one grid cell in the given direction |
| `PICK_UP` | Pick up the item at or immediately in front of the player |
| `DROP` | Place held item on the current cell or counter |
| `INTERACT` | Use the station at the current cell (chop, add to pot, serve dish) |
| `STAY` | Do nothing |

Both players submit actions simultaneously each step.

---

## Kitchen layouts

| `layout` | Description |
|---|---|
| `cramped_room` | Classic 5×5 layout; one pot, one onion source |
| `asymmetric_adv` | 6×6; ingredients near one player, serving near the other |
| `coordination_ring` | Ring; forces item passing across a central counter |

---

## Config keys (`env_kwargs`)

| Key | Default | Description |
|---|---|---|
| `num_players` | 2 | Currently only 2-player layouts are defined |
| `layout` | `cramped_room` | Kitchen layout name |
| `max_steps` | 400 | Episode length in steps |
| `cook_time` | 4 | Steps a pot must cook before dish is ready |
| `seed` | 0 | RNG seed (currently unused — layouts are deterministic) |

---

## Implementation status

| File | Status |
|---|---|
| `testbed/envs/symbolic/overcooked.py` | Layout definitions + state initialisation done; `submit()` grid simulation needed |
| `testbed/renderers/symbolic/overcooked.py` | Complete — renders kitchen state as structured text |
| `testbed/parsers/symbolic/overcooked.py` | Complete — parses MOVE/PICK_UP/DROP/INTERACT/STAY |
| `testbed/registry.py` | Lazy-loaded via `_SYMBOLIC_LAZY["overcooked"]` |

**To activate:** implement `submit()` in `testbed/envs/symbolic/overcooked.py`.
The docstring contains detailed pseudocode for each action branch.
Then un-comment `"overcooked"` in `testbed/registry.py`.

---

## Adding new layouts

Add an entry to `LAYOUTS` in `overcooked.py`:

```python
LAYOUTS["my_layout"] = {
    "grid": [
        list("WWWWW"),
        list("WO  W"),
        list("W P DW"),
        list("W   W"),
        list("WWWWW"),
    ],
    "player_starts": [(1, 1), (3, 3)],
    "description": "My custom layout description.",
}
```

Grid cell legend: `W`=wall, `O`=onion source, `T`=tomato source,
`P`=pot, `C`=chop board, `D`=delivery window, `S`=dish source, ` `=floor.

---

## Key research questions

- Does joint reasoning (ToM) improve coordination vs. independent planning?
- Which layout requires the most explicit role differentiation (one player cooks, one delivers)?
- Does a `contrarian` persona disrupt team coordination vs. a `cooperative` one?
- How does the text-based rendering fidelity affect action quality compared to grid-world RL baselines?

---

## Quick smoke test (after implementing submit)

```bash
python scripts/run_episode.py \
  --config config/games/overcooked/overcooked_noop_2p.yaml \
  --episodes 1
```
