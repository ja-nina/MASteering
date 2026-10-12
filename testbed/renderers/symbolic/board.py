from __future__ import annotations

from testbed.envs.symbolic.board import column_letter
from testbed.types import RawObs, RenderContext
from testbed.visual import BG, GREEN, PLAYER_COLORS, RED, WHITE, YELLOW, grid_board, text_center

_DISCS = [(RED, "red"), (YELLOW, "yellow"), (GREEN, "green"), ((240, 240, 240), "white")]  # on a navy board

_NAMES = {"tictactoe": "tic-tac-toe", "connect_four": "connect four", "gomoku": "gomoku"}


class BoardRenderer:
    modes = ("text", "image", "both")

    def system_prompt(self, agent_id: str, raw_obs: RawObs | None = None) -> str:
        obs = raw_obs or {}
        how = ("Choose a column; your mark drops to the lowest empty cell of that column."
               if obs.get("gravity") else "Choose any empty cell.")
        fmt = "MOVE: <column letter>" if obs.get("gravity") else "MOVE: <column letter><row number>, e.g. MOVE: B2"
        return (f"You are {agent_id} playing {_NAMES.get(obs.get('variant'), 'a board game')} on a "
                f"{obs.get('rows')}x{obs.get('cols')} board against {len(obs.get('marks', {})) - 1} opponent(s).\n\n"
                f"RULES\n- Players take turns placing their mark. {how}\n"
                f"- The first player with {obs.get('k')} marks in a row (horizontally, vertically or "
                "diagonally) wins. A full board is a draw.\n"
                "- Columns are letters (A, B, ...) from the left, rows are numbers from the top.\n\n"
                f"Respond in the form: {fmt}")

    def _whose(self, obs) -> str:
        marks = ", ".join(f"{pid} = {m}" for pid, m in obs["marks"].items())
        return f"Marks: {marks}. You are {obs['agent_id']} ({obs['my_mark']})."

    def _tail(self, obs) -> list:
        lines = []
        if obs["history"]:
            lines.append("Moves so far: " + ", ".join(f"{h['player']} {h['cell']}" for h in obs["history"]))
        if obs["gravity"]:
            lines.append("Columns you can play: " + ", ".join(column_letter(c) for _, c in obs["legal"]))
        lines.append("It is your turn. " + ("Respond in the form: MOVE: <column letter>" if obs["gravity"]
                                            else "Respond in the form: MOVE: <column letter><row number>"))
        return lines

    def render(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        obs = raw_obs
        header = "   " + " ".join(column_letter(c) for c in range(obs["cols"]))
        rows = [f"{r + 1:>2} " + " ".join(cell or "." for cell in row) for r, row in enumerate(obs["board"])]
        return "\n".join([self._whose(obs), "Board:", header, *rows, ""] + self._tail(obs))

    def render_visual(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> str:
        return "\n".join([self._whose(raw_obs), "The board is shown in the image.", ""] + self._tail(raw_obs))

    def render_images(self, raw_obs: RawObs, agent_id: str, context: RenderContext) -> list:
        obs = raw_obs
        if obs["gravity"]:
            color = {m: _DISCS[i][0] for i, m in enumerate(obs["marks"].values())}
            names = {m: _DISCS[i][1] for i, m in enumerate(obs["marks"].values())}
        else:
            color = {m: PLAYER_COLORS[i] for i, m in enumerate(obs["marks"].values())}
            names = {m: m for m in obs["marks"].values()}
        last = obs["history"][-1]["cell"] if obs["history"] else None
        cell = max(36, min(80, 480 // max(obs["rows"], obs["cols"])))

        def paint(draw, r, c, box):
            x0, y0, x1, y1 = box
            if obs["gravity"]:
                draw.rectangle(box, fill=(25, 45, 110))
                draw.ellipse([x0 + 5, y0 + 5, x1 - 5, y1 - 5], fill=color.get(obs["board"][r][c], BG))
            elif obs["board"][r][c]:
                m = obs["board"][r][c]
                text_center(draw, ((x0 + x1) / 2, (y0 + y1) / 2), m, int(cell * 0.6), color[m])
            if last == f"{column_letter(c)}{r + 1}":
                draw.rectangle(box, outline=WHITE if obs["gravity"] else YELLOW, width=4)

        marks = "  ".join(f"{pid}={names[m]}" for pid, m in obs["marks"].items())
        return [grid_board(obs["rows"], obs["cols"], cell, paint, header=f"{marks}   (framed = last move)",
                           row_labels=[str(r + 1) for r in range(obs["rows"])],
                           col_labels=[column_letter(c) for c in range(obs["cols"])])]
