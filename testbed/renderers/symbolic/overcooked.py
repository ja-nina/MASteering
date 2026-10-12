"""Renderer for Overcooked (text, or the kitchen as an image)."""
from __future__ import annotations

from testbed.types import RawObs, RenderContext
from testbed.visual import BLUE, GREEN, GREY, INK, PLAYER_COLORS, RED, YELLOW, grid_board, text_center

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
_CELL_FILL = {"W": (90, 90, 90), "O": (240, 220, 140), "T": (240, 150, 140), "P": (120, 120, 140),
              "C": (200, 170, 120), "D": (140, 200, 140), "S": (200, 220, 240), " ": (245, 242, 235)}


class OvercookedRenderer:
    modes = ("text", "image", "both")

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

    @staticmethod
    def _header(obs) -> str:
        return (f"=== Kitchen: {obs.get('layout_name', '?')} | Score: {obs.get('score', 0)} | "
                f"Step: {obs.get('step', 0)}/{obs.get('max_steps', '?')} ===")

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        obs = raw_obs
        lines = [self._header(obs), obs.get("layout_desc", ""), "",
                 f"Your position: {obs.get('my_pos', '?')}. Holding: {obs.get('my_held') or 'nothing'}."]
        for pid, info in obs.get("other_players", {}).items():
            lines.append(f"{pid}: position {info.get('pos', '?')}, holding: {info.get('held') or 'nothing'}.")
        lines.append("")
        pot_cooking = obs.get("pot_cooking", {})
        for pos_str, contents in obs.get("pot_contents", {}).items():
            if contents:
                cook_t = pot_cooking.get(pos_str, 0)
                status = (f"cooking ({cook_t} steps left)" if cook_t > 0
                          else "ready to serve" if len(contents) >= 3 else "needs more ingredients")
                lines.append(f"Pot at {pos_str}: {contents} — {status}.")
            else:
                lines.append(f"Pot at {pos_str}: empty.")
        floor_items = obs.get("items_on_floor", {})
        if floor_items:
            lines.append("")
            lines += [f"Item on floor at {pos_str}: {item}." for pos_str, item in floor_items.items()]
        return "\n".join(lines + self._tail(context))

    def render_visual(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        return "\n".join([self._header(raw_obs),
                          "The kitchen is shown in the image (rows and columns are numbered; North is up). "
                          "You are the circle marked YOU; a letter on a player is what they hold.", ""]
                         + self._tail(context))

    def render_images(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> list:
        obs = raw_obs
        grid = obs["grid"]
        players = {agent_id: (tuple(obs["my_pos"]), obs.get("my_held"))}
        players.update({pid: (tuple(i["pos"]), i.get("held")) for pid, i in obs.get("other_players", {}).items()})
        colors = {pid: (BLUE if pid == agent_id else PLAYER_COLORS[0]) for pid in players}
        pots, cooking, floor = obs.get("pot_contents", {}), obs.get("pot_cooking", {}), obs.get("items_on_floor", {})

        def paint(draw, r, c, box):
            x0, y0, x1, y1 = box
            kind = grid[r][c]
            draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], fill=_CELL_FILL.get(kind, (245, 242, 235)))
            if kind not in (" ", "W"):
                text_center(draw, (x0 + 10, y0 + 10), kind, 12, INK)
            key = str((r, c))
            if key in pots:
                n, t = len(pots[key] or []), cooking.get(key, 0)
                label = f"{n}/3" + (f" t{t}" if t else (" ready" if n >= 3 else ""))
                text_center(draw, ((x0 + x1) / 2, y1 - 10), label, 11, YELLOW if n >= 3 and not t else (255, 255, 255))
            if key in floor:
                text_center(draw, ((x0 + x1) / 2, (y0 + y1) / 2), str(floor[key])[:5], 11, RED)
            for pid, (pos, held) in players.items():
                if pos == (r, c):
                    draw.ellipse([x0 + 12, y0 + 12, x1 - 12, y1 - 12], fill=colors[pid], outline=INK, width=2)
                    text_center(draw, ((x0 + x1) / 2, (y0 + y1) / 2 - 6), "YOU" if pid == agent_id else pid.replace("player_", "P"), 11,
                                (255, 255, 255))
                    if held:
                        text_center(draw, ((x0 + x1) / 2, (y0 + y1) / 2 + 9), str(held)[:6], 11, GREEN if held else GREY)

        cell = max(60, min(96, 520 // max(len(grid), len(grid[0]))))
        legend = "  ".join(f"{k}={v}" for k, v in _CELL_LEGEND.items() if k.strip() and any(k in row for row in grid))
        return [grid_board(len(grid), len(grid[0]), cell, paint,
                           header=f"score {obs.get('score', 0)}  step {obs.get('step', 0)}/{obs.get('max_steps')}  {legend}")]

    @staticmethod
    def _tail(context: RenderContext) -> list:
        """Recent actions and the question (text in every observation mode)."""
        lines = []
        recent = [h for h in context.history if "actions" in h][-10:]
        if recent:
            lines.append("Recent actions:")
            for entry in recent:
                acts = ", ".join(f"{pid}: {a}" for pid, a in entry.get("actions", {}).items())
                lines.append(f"  Step {entry['step']}: {acts}{'  ← SCORED!' if entry.get('score_delta') else ''}")
            lines.append("")
        lines.append("What do you do next?")
        return lines
