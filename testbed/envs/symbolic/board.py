"""k-in-a-row board games: tic-tac-toe, connect four, gomoku (2+ players, turn-based).

Players take turns placing their mark; the first to get `k` marks in a row (horizontal,
vertical or diagonal) wins. With `gravity` (connect four) a mark drops to the lowest empty
cell of the chosen column. Columns are letters (A, B, ...), rows are numbers from the top.
The seed decides who moves first, so conditions see the same openings.

Rewards: winner 1, others 0; a full board is a draw (0.5 each).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from testbed.envs.symbolic.base import SymbolicAdapter
from testbed.types import Action, RawObs, StepResult

VARIANTS = {
    "tictactoe":    dict(rows=3, cols=3, k=3, gravity=False),
    "connect_four": dict(rows=6, cols=7, k=4, gravity=True),
    "gomoku":       dict(rows=9, cols=9, k=5, gravity=False),
}
MARKS = ["X", "O", "+", "#"]


def column_letter(c: int) -> str:
    return chr(ord("A") + c)


class BoardAdapter(SymbolicAdapter):
    def __init__(self, num_players: int = 2, variant: str = "tictactoe", seed: int = 0,
                 rows: Optional[int] = None, cols: Optional[int] = None, k: Optional[int] = None) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
        if not 2 <= num_players <= len(MARKS):
            raise ValueError(f"board games take 2-{len(MARKS)} players")
        super().__init__(num_players=num_players, num_rounds=10_000)
        v = dict(VARIANTS[variant])
        v.update({k_: x for k_, x in (("rows", rows), ("cols", cols), ("k", k)) if x is not None})
        self.variant, self.rows, self.cols, self.k, self.gravity = variant, v["rows"], v["cols"], v["k"], v["gravity"]
        self._seed = seed
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.context.extra.update({
            "board": [[None] * self.cols for _ in range(self.rows)],
            "current": random.Random(self._seed).randrange(self.num_players),
            "marks": {pid: MARKS[i] for i, pid in enumerate(self._ids)},
            "winner": None,
        })

    def legal_actions(self, agent_id: str) -> List[Tuple[int, int]]:
        board = self.context.extra["board"]
        if self.gravity:
            return [(self._drop_row(c), c) for c in range(self.cols) if board[0][c] is None]
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if board[r][c] is None]

    def _drop_row(self, c: int) -> int:
        board = self.context.extra["board"]
        return max(r for r in range(self.rows) if board[r][c] is None)

    def _observation(self, agent_id: str) -> RawObs:
        x = self.context.extra
        return {"agent_id": agent_id, "variant": self.variant, "rows": self.rows, "cols": self.cols,
                "k": self.k, "gravity": self.gravity, "board": [list(r) for r in x["board"]],
                "marks": dict(x["marks"]), "my_mark": x["marks"][agent_id],
                "current": self._ids[x["current"]], "legal": self.legal_actions(agent_id),
                "history": list(self.context.history)}

    def pending(self) -> List[Tuple[str, RawObs]]:
        pid = self._ids[self.context.extra["current"]]
        return [(pid, self._observation(pid))]

    def submit(self, actions: Dict[str, Action]) -> StepResult:
        x = self.context.extra
        pid = self._ids[x["current"]]
        legal = self.legal_actions(pid)
        move = tuple(actions[pid]) if isinstance(actions[pid], (list, tuple)) else None
        if move not in legal:  # parser fallback or a stale move: take the first legal cell
            move = legal[0]
        r, c = move
        x["board"][r][c] = x["marks"][pid]
        self.context.round_index += 1
        self.context.history.append({"turn": self.context.round_index, "player": pid,
                                     "cell": f"{column_letter(c)}{r + 1}"})
        if self._wins(r, c):
            x["winner"] = pid
            rewards = {p: float(p == pid) for p in self._ids}
            done = True
        elif not self.legal_actions(pid):
            rewards, done = {p: 0.5 for p in self._ids}, True
        else:
            rewards, done = {p: 0.0 for p in self._ids}, False
            x["current"] = (x["current"] + 1) % self.num_players
        self.context.last_rewards = rewards
        return StepResult(rewards=rewards, done=done,
                          info={"player": pid, "cell": f"{column_letter(c)}{r + 1}", "winner": x["winner"]})

    def _wins(self, r: int, c: int) -> bool:
        board, mark = self.context.extra["board"], self.context.extra["board"][r][c]
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            n = 1
            for sign in (1, -1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < self.rows and 0 <= cc < self.cols and board[rr][cc] == mark:
                    n += 1
                    rr, cc = rr + sign * dr, cc + sign * dc
            if n >= self.k:
                return True
        return False
