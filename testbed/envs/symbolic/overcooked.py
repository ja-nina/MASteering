"""Overcooked — cooperative multi-agent kitchen coordination (text-based).

Players work together to prepare and deliver dishes within a time limit.
The environment is a grid kitchen where players can move, pick up ingredients,
interact with stations (chopping board, pot, delivery window), and pass items.

Reference game: Carroll et al. (2019). On the Utility of Learning about Humans
for Human-AI Coordination. NeurIPS 2019.

This is a TEXT-BASED adaptation of the Overcooked kitchen domain, rendering
grid state as structured natural language for LLM agents.

Action space per player (simultaneous):
  MOVE <N|S|E|W>   — move one cell in the given direction
  PICK_UP           — pick up the item at the current cell or in front
  DROP              — drop held item at current cell
  INTERACT          — use the station at current cell (chop, add to pot, serve)
  STAY              — do nothing this step

Kitchen layouts (select via env_kwargs["layout"]):
  "cramped_room"    — 2 players, 5×4 grid, classic Overcooked layout
  "asymmetric_adv"  — 2 players, 5×4, asymmetric ingredient placement
  "coordination_ring" — 2 players, ring layout requiring item passing

Reward: +1 per dish successfully delivered to the serving window.

TODO: Implement the grid simulation.
The kitchen state should be stored in context.extra and rendered to text.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, RenderContext, StepResult

# ── Kitchen layout definitions ─────────────────────────────────────────────────
# Each cell is one of:
#   " " = floor, "W" = wall, "O" = onion source, "T" = tomato source,
#   "P" = pot station, "C" = chop board, "D" = delivery window, "S" = serving dish source
#
# Players start at marked positions; grid is (row, col) indexed.

LAYOUTS: Dict[str, Dict[str, Any]] = {
    "cramped_room": {
        "grid": [
            list("WWWWW"),
            list("WO  W"),
            list("W P DW"),  # note: pad to same width in real impl
            list("W   W"),
            list("WWWWW"),
        ],
        "player_starts": [(1, 1), (3, 1)],
        "description": "Small 5x5 room with one pot, one onion source, and delivery on the east wall.",
    },
    "asymmetric_adv": {
        "grid": [
            list("WWWWWW"),
            list("WO   W"),
            list("W  P W"),
            list("WC   W"),
            list("W    W"),
            list("WWDWWW"),
        ],
        "player_starts": [(1, 1), (4, 4)],
        "description": "Asymmetric layout: one player near ingredients, one near serving.",
    },
    "coordination_ring": {
        "grid": [
            list("WWWWW"),
            list("WO DW"),
            list("W   W"),
            list("WC PW"),
            list("WWWWW"),
        ],
        "player_starts": [(1, 1), (3, 3)],
        "description": "Ring layout: players must coordinate to pass items across the counter.",
    },
}

# Valid item types
ITEMS = {"onion", "tomato", "dish", "onion_soup", "tomato_soup", None}

# Recipes: {frozenset_of_ingredients → dish_name}
RECIPES: Dict[frozenset, str] = {
    frozenset(["onion", "onion", "onion"]): "onion_soup",
    frozenset(["tomato", "tomato", "tomato"]): "tomato_soup",
}


class OvercookedAdapter(SymbolicAdapter):
    """Text-based Overcooked environment.

    Simultaneous: all players act every step (pending() returns all players).

    State stored in context.extra:
      "grid"          : 2D list of cell types (mutable: items placed on floor)
      "player_pos"    : {player_id: (row, col)}
      "player_held"   : {player_id: item_name or None}
      "pot_contents"  : {(row, col): [item_name, ...]}  — items in each pot
      "pot_cooking"   : {(row, col): int}               — remaining cook steps (0 = done)
      "items_on_floor": {(row, col): item_name}
      "score"         : int — dishes delivered
      "step"          : int

    TODO: Implement _observation() and submit() with the grid simulation.
    """

    DIRECTIONS = {
        "N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1),
    }

    def __init__(
        self,
        num_players: int = 2,
        layout: str = "cramped_room",
        max_steps: int = 400,
        cook_time: int = 4,   # steps in pot before dish is ready
        seed: int = 0,
    ) -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown layout {layout!r}. Choose from: {list(LAYOUTS)}")
        super().__init__(num_players=num_players, num_rounds=max_steps)
        self._layout_name = layout
        self._layout      = LAYOUTS[layout]
        self._cook_time   = cook_time
        self._seed        = seed
        self._max_steps   = max_steps

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset()
        import copy
        grid  = copy.deepcopy(self._layout["grid"])
        starts = self._layout["player_starts"]

        player_pos  = {pid: tuple(starts[i]) for i, pid in enumerate(self._ids)}
        player_held = {pid: None for pid in self._ids}

        # Find pot positions in grid
        pots: List[Tuple[int, int]] = []
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell == "P":
                    pots.append((r, c))

        self.context.extra.update({
            "grid":           grid,
            "layout_desc":    self._layout["description"],
            "player_pos":     player_pos,
            "player_held":    player_held,
            "pot_contents":   {pos: [] for pos in pots},
            "pot_cooking":    {pos: 0  for pos in pots},
            "items_on_floor": {},
            "score":          0,
            "step":           0,
        })

    # ── observation ───────────────────────────────────────────────────────────

    def _observation(self, agent_id: str) -> RawObs:
        """Return a text-ready kitchen snapshot for agent_id.

        TODO: Render the grid state as a structured text description
        that the LLM can understand and reason over.

        Suggested format:
          Kitchen (cramped_room): Small 5x5 room...
          Your position: (2, 3). Holding: onion.
          Other players: player_1 at (1, 1), holding: nothing.
          Pot at (2,2): [onion, onion] — needs 1 more ingredient; not cooking.
          Onion source at (1,1). Delivery window at (2,4).
          Score: 3 dishes delivered. Step 42/400.
        """
        extra = self.context.extra
        return {
            "agent_id":       agent_id,
            "step":           extra["step"],
            "max_steps":      self._max_steps,
            "layout_name":    self._layout_name,
            "layout_desc":    extra["layout_desc"],
            "my_pos":         extra["player_pos"][agent_id],
            "my_held":        extra["player_held"][agent_id],
            "other_players":  {
                pid: {"pos": extra["player_pos"][pid], "held": extra["player_held"][pid]}
                for pid in self._ids if pid != agent_id
            },
            "pot_contents":   {str(pos): contents for pos, contents in extra["pot_contents"].items()},
            "pot_cooking":    {str(pos): t for pos, t in extra["pot_cooking"].items()},
            "items_on_floor": {str(pos): item for pos, item in extra["items_on_floor"].items()},
            "score":          extra["score"],
            "grid":           extra["grid"],
        }

    # ── submit ─────────────────────────────────────────────────────────────────

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        """Execute all players' actions simultaneously and advance the kitchen state.

        Expected action formats (see OvercookedParser):
          "MOVE N" | "MOVE S" | "MOVE E" | "MOVE W"
          "PICK_UP"
          "DROP"
          "INTERACT"
          "STAY"

        TODO: implement the grid simulation.

        Pseudocode:
          1. Process MOVE actions (check grid bounds + walls + player collisions).
          2. Process PICK_UP: pick up adjacent or same-cell item; place in player_held.
          3. Process DROP: place held item on current cell (if floor/counter).
          4. Process INTERACT:
             - At onion/tomato source: pick up ingredient.
             - At pot: add held ingredient; if pot full → start cooking.
             - At dish source: pick up empty dish.
             - At chop board: chop held ingredient (some recipes need chopped items).
             - At delivery: if holding completed dish → score += 1.
          5. Advance pot_cooking timers; pots that hit 0 → dish ready.
          6. Advance step; check done (step >= max_steps).
          7. Reward = change in score this step.
        """
        extra = self.context.extra
        raise NotImplementedError(
            "OvercookedAdapter.submit() not yet implemented. "
            "See docstring for the expected grid simulation logic."
        )

    def close(self) -> Dict[str, float]:
        return dict(self.context.last_rewards)
