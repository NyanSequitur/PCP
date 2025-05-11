from __future__ import annotations

"""Utility functions and basic rules for a simple Connect‑Four‑like game.

The module implements a 7 × 6 board (columns × rows) where the *lowest* row
(index 0) is the one in which new pieces are inserted first.  Players drop a
piece into a column; the piece occupies the first empty cell in that column.

The public helpers exposed here are **pure** (except for
``apply_player_action`` which mutates its *board* argument in place) and do not
print or read anything.  They are therefore easy to test and reuse.
"""

from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Board constants
# ──────────────────────────────────────────────────────────────────────────────

BOARD_COLS: int = 7
BOARD_ROWS: int = 6
BOARD_SHAPE: tuple[int, int] = (BOARD_ROWS, BOARD_COLS)
INDEX_HIGHEST_ROW: int = BOARD_ROWS - 1
INDEX_LOWEST_ROW: int = 0

BoardPiece = np.int8  # dtype used to store pieces in the numpy array
NO_PLAYER: BoardPiece = BoardPiece(0)
PLAYER1: BoardPiece = BoardPiece(1)
PLAYER2: BoardPiece = BoardPiece(2)

# Symbols used when pretty‑printing the board
BoardPiecePrint = str
NO_PLAYER_PRINT: BoardPiecePrint = " "
PLAYER1_PRINT: BoardPiecePrint = "X"
PLAYER2_PRINT: BoardPiecePrint = "O"

# The *action* that an agent returns is the column index.  Using the same dtype
# as we do in the board makes comparisons cheaper when the board is stored in
# the same integer size.
PlayerAction = np.int8


class GameState(Enum):
    """Enumeration describing the state from the perspective of *the player who
    has just moved*.
    """

    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class MoveStatus(Enum):
    """Why a prospective move is valid or not."""

    IS_VALID = 1
    WRONG_TYPE = "Input does not have the correct type (PlayerAction)."
    OUT_OF_BOUNDS = "Input is out of bounds."
    FULL_COLUMN = "Selected column is full."


class SavedState:  # pragma: no cover – placeholder for agents
    pass


# Type signature for an *agent* callable that is compatible with the
# autograder/competition harness (kept for completeness; not used in this
# module itself).
GenMove = Callable[[np.ndarray, BoardPiece, Optional[SavedState]], tuple[PlayerAction, Optional[SavedState]]]

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def initialize_game_state() -> np.ndarray:
    """Return a fresh, empty board of shape ``BOARD_SHAPE`` and dtype
    :class:`BoardPiece` (numpy ``int8``).
    """
    return np.zeros(BOARD_SHAPE, dtype=BoardPiece)


# Helper map between internal int representation and printable character
_PRINT_MAP: dict[int, str] = {
    int(NO_PLAYER): NO_PLAYER_PRINT,
    int(PLAYER1): PLAYER1_PRINT,
    int(PLAYER2): PLAYER2_PRINT,
}
_REVERSE_PRINT_MAP: dict[str, BoardPiece] = {
    NO_PLAYER_PRINT: NO_PLAYER,
    PLAYER1_PRINT: PLAYER1,
    PLAYER2_PRINT: PLAYER2,
}


def pretty_print_board(board: np.ndarray) -> str:  # noqa: C901 – complexity fine for a utility
    """Return a *multi‑line* human readable string representation of *board*.

    The lowest row of *board* (index 0) is shown **at the bottom** of the
    returned string so the printed board looks the same orientation that a
    player sees.
    """
    if board.shape != BOARD_SHAPE:
        raise ValueError(f"Board must have shape {BOARD_SHAPE}, got {board.shape}.")

    # Build row strings from *top* to *bottom*
    row_lines: list[str] = []
    for row in range(BOARD_ROWS - 1, -1, -1):  # highest → lowest
        cells = "".join(f"{_PRINT_MAP[int(board[row, col])]} " for col in range(BOARD_COLS))
        row_lines.append(f"|{cells}|")

    horizontal_border = f"|{'=' * (BOARD_COLS * 2)}|"
    index_line = "|" + " ".join(str(col) for col in range(BOARD_COLS)) + " |"

    return "\n".join([horizontal_border, *row_lines, horizontal_border, index_line])


def string_to_board(pp_board: str) -> np.ndarray:  # noqa: C901 – linear parser, acceptable complexity
    """Convert the *exact* output of :func:`pretty_print_board` back into a
    numpy board.

    This is handy when an agent crashes and leaves a board snapshot in the log
    that needs to be reconstructed programmatically while debugging.
    """
    lines = pp_board.splitlines()
    if len(lines) != BOARD_ROWS + 3:
        raise ValueError("Input does not have the correct number of lines.")

    # Validate top & bottom borders + index line structure quickly.
    expected_border = f"|{'=' * (BOARD_COLS * 2)}|"
    if lines[0] != expected_border or lines[-2] != expected_border:
        raise ValueError("Input does not contain the expected borders – did the format change?")

    if not lines[-1].startswith("|0 1 2"):
        raise ValueError("Could not detect index line at the bottom of the board string.")

    board = initialize_game_state()

    # Parse *printable* rows, skipping top border and bottom border + indices.
    for printable_row_index, printable_row in enumerate(lines[1:-2]):
        if len(printable_row) != BOARD_COLS * 2 + 2:
            raise ValueError("Row length mismatch while parsing board string.")

        # Row 0 in *pp_board* is the *highest* row, hence subtract from BOARD_ROWS‑1.
        board_row = BOARD_ROWS - 1 - printable_row_index
        for col in range(BOARD_COLS):
            # Cell character is at offset 1 + 2*col inside the printable row.
            char = printable_row[1 + 2 * col]
            try:
                board[board_row, col] = _REVERSE_PRINT_MAP[char]
            except KeyError as exc:  # pragma: no cover – helps debugging
                raise ValueError(f"Unexpected character '{char}' while parsing board string.") from exc

    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """Drop *player*'s piece into *action* column, modifying *board* **in place**.

    The piece occupies the *lowest* empty row in that column.  Raises
    :class:`ValueError` if the column is full or *action* is out of bounds.
    """
    if not 0 <= int(action) < BOARD_COLS:
        raise ValueError("'action' out of bounds.")

    # Find the first empty row (the *lowest* one) in the selected column.
    for row in range(BOARD_ROWS):
        if board[row, action] == NO_PLAYER:
            board[row, action] = player
            return  # done
    raise ValueError("Selected column is full – cannot apply action.")


# Shifts used to detect connections in the board for each direction (dx, dy)
_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),   # horizontal
    (0, 1),   # vertical
    (1, 1),   # diagonal ↗︎
    (1, -1),  # diagonal ↘︎
)


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:  # noqa: C901 – nested loops are the clearest implementation
    """Return *True* iff *player* has 4 connected pieces in *board*.

    The check is performed in all four directions: horizontal, vertical and
    the two diagonals.  Runs in **O(BOARD_ROWS × BOARD_COLS)** which is fine
    for such a small board.
    """
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row, col] != player:
                continue

            # Check all four directions from this piece
            for dx, dy in _DIRECTIONS:
                count = 1
                for step in range(1, 4):  # check next three pieces
                    new_row = row + dy * step
                    new_col = col + dx * step

                    if not (0 <= new_row < BOARD_ROWS and 0 <= new_col < BOARD_COLS):
                        break

                    if board[new_row, new_col] == player:
                        count += 1
                    else:
                        break

                if count == 4:
                    return True

    return False


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """Evaluate the state of the game from *player*'s perspective **after they
    have just moved**.
    """
    if connected_four(board, player):
        return GameState.IS_WIN

    if np.all(board != NO_PLAYER):
        return GameState.IS_DRAW

    return GameState.STILL_PLAYING


def _is_player_action(value: Any) -> bool:
    """Return True iff *value* is exactly of type :class:`numpy.int8`.  We avoid
    accepting Python ``int`` here so that tests catch accidental type changes
    early.
    """
    return isinstance(value, np.int8)


def check_move_status(board: np.ndarray, column: Any) -> MoveStatus:  # noqa: C901 – explicit if/else chain increases clarity
    """Validate whether a proposed *column* is a legal move on *board*.

    The function performs *static* validation only and **does not** mutate the
    board.
    """
    if not _is_player_action(column):
        return MoveStatus.WRONG_TYPE

    col_index = int(column)

    if not 0 <= col_index < BOARD_COLS:
        return MoveStatus.OUT_OF_BOUNDS

    if board[INDEX_HIGHEST_ROW, col_index] != NO_PLAYER:
        return MoveStatus.FULL_COLUMN

    return MoveStatus.IS_VALID


def test_check_end_state_draw():
    board = initialize_game_state()
    
    # Fill the board with a known draw pattern (alternating X and O)
    draw_pattern = [
        [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],  # Top row
        [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
        [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
        [PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2],
        [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
        [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],  # Bottom row
    ]

    for row_idx, row in enumerate(draw_pattern):
        for col_idx, piece in enumerate(row):
            board[BOARD_ROWS - 1 - row_idx, col_idx] = piece

    assert check_end_state(board, PLAYER1) == GameState.IS_DRAW
    assert check_end_state(board, PLAYER2) == GameState.IS_DRAW
