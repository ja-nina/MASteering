"""Renderer for text-based Overcooked."""
from __future__ import annotations

from testbed.types import RawObs, RenderContext

_CELL_LEGEND = {
    "W": "wall",
    "O": "onion source",
    "T": "tomato source",
    "P": "pot station",
    "C": "chop board",
    "D": "delivery window",
    "S": "dish source",
    " ": "floor",
}


class OvercookedRenderer:
    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        return "\n".join([
            "You are playing a cooperative kitchen game (Overcooked-style).",
            "",
            "OBJECTIVE: Prepare and deliver as many dishes as possible before time runs out.",
            "You and your partner must coordinate — one player often needs to pass ingredients to the other.",
            "",
            "ACTIONS (one per step, simultaneous with your partner):",
            "  MOVE N  — move one step North",
            "  MOVE S  — move one step South",
            "  MOVE E  — move one step East",
            "  MOVE W  — move one step West",
            "  PICK_UP — pick up the item at your current position or the adjacent counter",
            "  DROP    — drop your held item at the current position / onto the counter",
            "  INTERACT — use the station at your position (chop, add to pot, serve dish)",
            "  STAY    — do nothing this step",
            "",
            "RECIPE: Place 3 ingredients in the pot → wait for it to cook → pick up dish → deliver to window.",
            "Respond with exactly one action per step.",
        ])

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        obs   = raw_obs
        step  = obs.get("step", 0)
        mstep = obs.get("max_steps", "?")
        score = obs.get("score", 0)
        pos   = obs.get("my_pos", "?")
        held  = obs.get("my_held") or "nothing"

        lines = [
            f"=== Kitchen: {obs.get('layout_name', '?')} | Score: {score} | Step: {step}/{mstep} ===",
            obs.get("layout_desc", ""),
            "",
            f"Your position: {pos}. Holding: {held}.",
        ]

        # Other players
        for pid, info in obs.get("other_players", {}).items():
            other_held = info.get("held") or "nothing"
            lines.append(f"{pid}: position {info.get('pos', '?')}, holding: {other_held}.")
        lines.append("")

        # Pot states
        pot_contents = obs.get("pot_contents", {})
        pot_cooking  = obs.get("pot_cooking",  {})
        for pos_str, contents in pot_contents.items():
            if contents:
                cook_t = pot_cooking.get(pos_str, 0)
                status = f"cooking ({cook_t} steps left)" if cook_t > 0 else "ready to serve" if len(contents) >= 3 else "needs more ingredients"
                lines.append(f"Pot at {pos_str}: {contents} — {status}.")
            else:
                lines.append(f"Pot at {pos_str}: empty.")

        # Floor items
        floor_items = obs.get("items_on_floor", {})
        if floor_items:
            lines.append("")
            for pos_str, item in floor_items.items():
                lines.append(f"Item on floor at {pos_str}: {item}.")

        lines.append("")
        lines.append("What do you do next?")
        return "\n".join(lines)
