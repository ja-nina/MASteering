"""Overcooked — cooperative multi-agent kitchen coordination (text-based).

Players work together to prepare and deliver dishes within a time limit.
The environment is a grid kitchen; players move, pick up ingredients,
interact with stations (pot, delivery window), and pass items.

Action space per player (simultaneous):
  MOVE <N|S|E|W>   — move one cell in the given direction
  PICK_UP           — pick up item lying on the floor at current cell
  DROP              — drop held item on current cell
  INTERACT          — use the nearest adjacent station
  STAY              — do nothing

Grid cell types:
  " " = walkable floor   "W" = wall (impassable)
  "O" = onion dispenser  "T" = tomato dispenser
  "P" = pot              "D" = delivery window
  "C" = chop board       "S" = dish source

Recipe implemented: 3 onions in a pot → cook cook_time steps → onion_soup.
Pick soup from pot, walk to delivery, INTERACT to score +1.

Reference: Carroll et al. (2019). On the Utility of Learning about Humans
for Human-AI Coordination. NeurIPS 2019.
"""
from __future__ import annotations

import copy
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, RenderContext, StepResult

# ── Layout definitions (all rows must have equal length) ──────────────────────

LAYOUTS: Dict[str, Dict[str, Any]] = {
    "cramped_room": {
        # 5 cols × 5 rows — onion north-west, pot centre, delivery south
        "grid": [
            list("WWWWW"),
            list("WO  W"),
            list("W P W"),
            list("W   W"),
            list("WWDWW"),
        ],
        "player_starts": [(1, 2), (3, 2)],
        "description": "Cramped 5×5 kitchen. Onion dispenser north-west, pot centre, delivery south.",
    },
    "asymmetric_adv": {
        # 6 cols × 6 rows — ingredients north, pot mid, delivery south
        "grid": [
            list("WWWWWW"),
            list("WO   W"),
            list("W    W"),
            list("W  P W"),
            list("W    W"),
            list("WWDWWW"),
        ],
        "player_starts": [(1, 2), (4, 2)],
        "description": "Asymmetric: one player near ingredients (north), one near serving (south).",
    },
    "coordination_ring": {
        # 5 cols × 5 rows — onion north-west, delivery north-east, pot south
        "grid": [
            list("WWWWW"),
            list("WO DW"),
            list("W   W"),
            list("W P W"),
            list("WWWWW"),
        ],
        "player_starts": [(1, 2), (2, 1)],
        "description": "Ring layout: players must pass items across the kitchen to coordinate.",
    },
}


class OvercookedAdapter(SymbolicAdapter):
    """Text-based Overcooked.  All players act every step (simultaneous)."""

    DIRECTIONS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

    def __init__(
        self,
        num_players: int = 2,
        layout: str = "cramped_room",
        max_steps: int = 400,
        cook_time: int = 4,
        seed: int = 0,
    ) -> None:
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown layout {layout!r}; choose from {list(LAYOUTS)}")
        super().__init__(num_players=num_players, num_rounds=max_steps)
        self._layout_name = layout
        self._layout      = LAYOUTS[layout]
        self._cook_time   = cook_time
        self._max_steps   = max_steps
        self._seed        = seed

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset()
        effective = seed if seed is not None else self._seed
        rng = random.Random(effective)

        grid = copy.deepcopy(self._layout["grid"])

        # Randomise starting positions over walkable floor cells so each
        # episode presents agents with a different initial configuration.
        floor_cells = [
            (r, c)
            for r, row in enumerate(grid)
            for c, cell in enumerate(row)
            if cell == " "
        ]
        if len(floor_cells) >= self.num_players:
            starts = rng.sample(floor_cells, self.num_players)
        else:
            starts = list(self._layout["player_starts"])

        player_pos  = {pid: tuple(starts[i]) for i, pid in enumerate(self._ids)}
        player_held = {pid: None for pid in self._ids}

        pots: List[Tuple[int, int]] = [
            (r, c)
            for r, row in enumerate(grid)
            for c, cell in enumerate(row)
            if cell == "P"
        ]

        self.context.extra.update({
            "grid":           grid,
            "layout_desc":    self._layout["description"],
            "player_pos":     player_pos,
            "player_held":    player_held,
            "pot_contents":   {pos: []      for pos in pots},
            "pot_cooking":    {pos: 0       for pos in pots},
            "pot_status":     {pos: "idle"  for pos in pots},
            "items_on_floor": {},
            "score":          0,
            "step":           0,
        })

    # ── helpers ────────────────────────────────────────────────────────────────

    def _find_adjacent_station(
        self, grid: List[List[str]], r: int, c: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        """Return (pos, cell_type) of the first adjacent non-floor, non-wall cell."""
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]):
                cell = grid[nr][nc]
                if cell not in (" ", "W"):
                    return (nr, nc), cell
        return None, None

    # ── observation ───────────────────────────────────────────────────────────

    def _observation(self, agent_id: str) -> RawObs:
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
            "pot_contents":   {str(pos): c for pos, c in extra["pot_contents"].items()},
            "pot_cooking":    {str(pos): t for pos, t in extra["pot_cooking"].items()},
            "pot_status":     {str(pos): s for pos, s in extra["pot_status"].items()},
            "items_on_floor": {str(pos): item for pos, item in extra["items_on_floor"].items()},
            "score":          extra["score"],
            "grid":           extra["grid"],
        }

    def pending(self) -> List[Tuple[str, RawObs]]:
        return [(pid, self._observation(pid)) for pid in self._ids]

    # ── submit ─────────────────────────────────────────────────────────────────

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        extra = self.context.extra
        grid  = extra["grid"]

        # ── 1. Resolve MOVE actions ──────────────────────────────────────────
        old_pos  = dict(extra["player_pos"])
        desired  = {}
        for pid in self._ids:
            act = str(actions.get(pid, "STAY")).upper()
            r, c = old_pos[pid]
            if act.startswith("MOVE ") and len(act) > 5:
                dr, dc = self.DIRECTIONS.get(act[5], (0, 0))
                nr, nc = r + dr, c + dc
                rows = len(grid)
                cols = len(grid[0]) if rows > 0 else 0
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == " ":
                    desired[pid] = (nr, nc)
                else:
                    desired[pid] = (r, c)
            else:
                desired[pid] = (r, c)

        # Block: multiple players want the same cell
        want_count = Counter(desired.values())
        new_pos    = {
            pid: (desired[pid] if want_count[desired[pid]] == 1 else old_pos[pid])
            for pid in self._ids
        }

        # Block: swaps (A→B's old cell, B→A's old cell simultaneously)
        ids = self._ids
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                p1, p2 = ids[i], ids[j]
                if desired[p1] == old_pos[p2] and desired[p2] == old_pos[p1]:
                    new_pos[p1] = old_pos[p1]
                    new_pos[p2] = old_pos[p2]

        extra["player_pos"] = new_pos

        # ── 2. Non-move actions (processed in player order) ──────────────────
        score_delta = 0

        for pid in self._ids:
            act  = str(actions.get(pid, "STAY")).upper()
            r, c = extra["player_pos"][pid]
            held = extra["player_held"][pid]

            if act == "PICK_UP":
                pos_key = (r, c)
                if held is None and pos_key in extra["items_on_floor"]:
                    extra["player_held"][pid] = extra["items_on_floor"].pop(pos_key)

            elif act == "DROP":
                if held is not None:
                    pos_key = (r, c)
                    if pos_key not in extra["items_on_floor"]:
                        extra["items_on_floor"][pos_key] = held
                        extra["player_held"][pid] = None

            elif act == "INTERACT":
                station_pos, stype = self._find_adjacent_station(grid, r, c)
                if station_pos is None:
                    continue

                sr, sc = station_pos

                if stype == "O":  # onion dispenser
                    if held is None:
                        extra["player_held"][pid] = "onion"

                elif stype == "T":  # tomato dispenser
                    if held is None:
                        extra["player_held"][pid] = "tomato"

                elif stype == "P":  # pot
                    pot_key    = (sr, sc)
                    contents   = extra["pot_contents"][pot_key]
                    pot_status = extra["pot_status"][pot_key]

                    if pot_status == "ready" and held is None:
                        # Collect the finished soup
                        soup = "onion_soup" if all(i == "onion" for i in contents) else "soup"
                        extra["player_held"][pid]     = soup
                        extra["pot_contents"][pot_key] = []
                        extra["pot_status"][pot_key]   = "idle"
                        extra["pot_cooking"][pot_key]  = 0

                    elif pot_status == "idle" and held in ("onion", "tomato"):
                        contents.append(held)
                        extra["player_held"][pid] = None
                        if len(contents) >= 3:
                            extra["pot_status"][pot_key]  = "cooking"
                            extra["pot_cooking"][pot_key] = self._cook_time

                elif stype == "D":  # delivery window
                    if held in ("onion_soup", "tomato_soup", "soup"):
                        extra["player_held"][pid] = None
                        extra["score"] += 1
                        score_delta    += 1

                elif stype == "S":  # dish source — no dish needed in this recipe
                    if held is None:
                        extra["player_held"][pid] = "dish"

        # ── 3. Advance pot timers ────────────────────────────────────────────
        for pot_key in extra["pot_cooking"]:
            if extra["pot_status"][pot_key] == "cooking":
                extra["pot_cooking"][pot_key] -= 1
                if extra["pot_cooking"][pot_key] <= 0:
                    extra["pot_status"][pot_key]  = "ready"
                    extra["pot_cooking"][pot_key]  = 0

        # ── 4. Advance step ──────────────────────────────────────────────────
        step_just_done = extra["step"]
        extra["step"]          += 1
        self.context.round_index += 1

        self.context.history.append({
            "step":        step_just_done,
            "actions":     {pid: str(actions.get(pid, "STAY")) for pid in self._ids},
            "score_delta": score_delta,
            "score":       extra["score"],
        })

        done    = extra["step"] >= self._max_steps
        rewards = {pid: float(score_delta) for pid in self._ids}

        return StepResult(
            rewards=rewards,
            done=done,
            info={"score": extra["score"], "step": extra["step"]},
        )

    def close(self) -> Dict[str, float]:
        return dict(self.context.last_rewards)
