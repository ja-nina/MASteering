# Board games: tic-tac-toe, connect four, gomoku

**Family:** `symbolic` | **IDs:** `tictactoe` (3×3, 3 in a row), `connect_four` (6×7, 4 in a row,
gravity), `gomoku` (9×9, 5 in a row)

One k-in-a-row engine (`testbed/envs/symbolic/board.py`) with 2–4 players taking turns. The seed
decides who moves first. Columns are letters, rows are numbers from the top.

**Action:** `MOVE: B2` (`MOVE: D` for connect four). Illegal moves are re-asked, listing the legal ones.
**Rewards:** winner 1, others 0; a full board is a draw (0.5 each).

**Observation modes:** `text` (an ASCII board), `image` (a drawn board with labelled rows and
columns, and the last move framed), or `both`.

## Config keys (`env_kwargs`)

| Key | Default | Description |
|---|---|---|
| `rows`, `cols`, `k` | from the variant | Override the board size or the winning length |

## Research angle

These are the cleanest place to test whether persona steering survives an image in context:
the rules are simple, the outcome is objective, and the same game can be played with text and
with image observations.
