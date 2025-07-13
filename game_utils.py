"""
Utility functions and basic rules for a simple Connect‑Four‑like game.

The module implements a 7 × 6 board (columns × rows) where the *lowest* row
(index 0) is the one in which new pieces are inserted first.  Players drop a
piece into a column; the piece occupies the first empty cell in that column.

The public helpers exposed here are **pure** (except for
``apply_player_action`` which mutates its *board* argument in place) and do not
print or read anything.  They are therefore easy to test and reuse.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Optional
import os
import sys

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
    """
    Enumeration describing the state from the perspective of the player who has just moved.

    Attributes
    ----------
    IS_WIN : int
        Indicates the player has won.
    IS_DRAW : int
        Indicates the game is a draw.
    STILL_PLAYING : int
        Indicates the game is still ongoing.
    """

    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class MoveStatus(Enum):
    """
    Enumeration describing why a prospective move is valid or not.

    Attributes
    ----------
    IS_VALID : int
        Move is valid.
    WRONG_TYPE : str
        Input does not have the correct type (PlayerAction).
    OUT_OF_BOUNDS : str
        Input is out of bounds.
    FULL_COLUMN : str
        Selected column is full.
    """

    IS_VALID = 1
    WRONG_TYPE = "Input does not have the correct type (PlayerAction)."
    OUT_OF_BOUNDS = "Input is out of bounds."
    FULL_COLUMN = "Selected column is full."


class SavedState:
    """
    Placeholder for agent state to persist between moves.
    """

    pass


# Type signature for an *agent* callable that is compatible with the
# autograder/competition harness (kept for completeness; not used in this
# module itself).
GenMove = Callable[[np.ndarray, BoardPiece, Optional[SavedState]], tuple[PlayerAction, Optional[SavedState]]]

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def initialize_game_state() -> np.ndarray:
    """
    Return a fresh, empty board of shape BOARD_SHAPE and dtype BoardPiece (numpy int8).

    Returns
    -------
    np.ndarray
        An empty game board.
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


def clear_console():
    """
    Clear the console screen in a cross-platform way.
    """
    if sys.platform.startswith("win"):
        os.system("cls")
    else:
        os.system("clear")


def user_col_to_internal(col: int) -> int:
    """
    Convert user-facing column number (1-7) to internal index (0-6).
    """
    return col - 1


def internal_col_to_user(col: int) -> int:
    """
    Convert internal column index (0-6) to user-facing number (1-7).
    """
    return col + 1


def user_row_to_internal(row: int) -> int:
    """
    Convert user-facing row number (1-6) to internal index (0-5).
    """
    return row - 1


def internal_row_to_user(row: int) -> int:
    """
    Convert internal row index (0-5) to user-facing number (1-6).
    """
    return row + 1


def pretty_print_board(board: np.ndarray) -> str:
    """
    Return a multi-line human readable string representation of board.

    The lowest row of board (index 0) is shown at the bottom of the returned string so the printed board looks the same orientation that a player sees.

    Parameters
    ----------
    board : np.ndarray
        The game board.

    Returns
    -------
    str
        Human-readable board string.
    """
    clear_console()
    if board.shape != BOARD_SHAPE:
        raise ValueError(f"Board must have shape {BOARD_SHAPE}, got {board.shape}.")

    # Build row strings from *top* to *bottom*
    row_lines: list[str] = []
    for row_index, row_data in enumerate(board[::-1, :]):  # highest → lowest
        cells = "".join(f" {_PRINT_MAP[int(col)]} " for col in row_data)
        # Calculate actual row index for user-facing display
        actual_row = BOARD_ROWS - 1 - row_index
        # Show user-facing row number (1-based)
        row_lines.append(f"|{cells}|  ← {internal_row_to_user(actual_row)}")

    horizontal_border = f"|{'=' * (BOARD_COLS * 3)}|"
    # Show user-facing column numbers (1-7)
    index_line = "|" + "".join(f" {internal_col_to_user(col)} " for col in range(BOARD_COLS)) + "|"

    legend = f"\nLegend: {PLAYER1_PRINT} = Player 1, {PLAYER2_PRINT} = Player 2\n"
    return "\n".join([
        legend,
        horizontal_border,
        *row_lines,
        horizontal_border,
        index_line
    ])


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Convert the exact output of pretty_print_board back into a numpy board.

    Parameters
    ----------
    pp_board : str
        Board string as produced by pretty_print_board.

    Returns
    -------
    np.ndarray
        The reconstructed game board.
    """
    lines = pp_board.splitlines()
    # pretty_print_board outputs:
    # 0: blank line
    # 1: legend
    # 2: blank line
    # 3: horizontal border
    # 4-9: board rows (top to bottom)
    # 10: horizontal border
    # 11: index line
    if len(lines) != BOARD_ROWS + 6:  # 6 rows + 3 blank/legend + 2 borders + index = 12
        raise ValueError(f"Input does not have the correct number of lines. Got {len(lines)} lines: {lines}")

    horizontal_border = f"|{'=' * (BOARD_COLS * 3)}|"
    if lines[3] != horizontal_border or lines[-2] != horizontal_border:
        raise ValueError("Input does not contain the expected borders – did the format change?")

    expected_index_line = "|" + "".join(f" {internal_col_to_user(col)} " for col in range(BOARD_COLS)) + "|"
    if lines[-1] != expected_index_line:
        raise ValueError("Could not detect index line at the bottom of the board string.")

    board = initialize_game_state()

    # Parse *printable* rows, skipping blank, legend, blank, top border, and bottom border + indices.
    for printable_row_index, printable_row in enumerate(lines[4:-2]):
        # Row 0 in *pp_board* is the *highest* row, hence subtract from BOARD_ROWS‑1.
        board_row = BOARD_ROWS - 1 - printable_row_index
        # Each row line: | X  O  X  ... |  ← 6
        # Extract cells between the first and last '|'
        row_content = printable_row.split('|')[1]
        cell_chars = [row_content[i*3+1] for i in range(BOARD_COLS)]
        for col in range(BOARD_COLS):
            char = cell_chars[col]
            try:
                board[board_row, col] = _REVERSE_PRINT_MAP[char]
            except KeyError as exc:
                raise ValueError(f"Unexpected character '{char}' while parsing board string.") from exc

    return board


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """
    Drop player's piece into action column, modifying board in place.

    The piece occupies the lowest empty row in that column. Raises ValueError if the column is full or action is out of bounds.

    Parameters
    ----------
    board : np.ndarray
        The game board.
    action : PlayerAction
        The column to drop the piece into.
    player : BoardPiece
        The player making the move.
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


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Return True iff player has 4 connected pieces in board.

    The check is performed in all four directions: horizontal, vertical and the two diagonals. Runs in O(BOARD_ROWS × BOARD_COLS).

    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to check for a win.

    Returns
    -------
    bool
        True if player has four in a row, else False.
    """
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row, col] != player:
                continue  # Skip cells not occupied by the player

            # Check all four directions from this piece
            for dx, dy in _DIRECTIONS:
                count = 1
                for step in range(1, 4):  # check next three pieces
                    new_row = row + dy * step
                    new_col = col + dx * step

                    if not (0 <= new_row < BOARD_ROWS and 0 <= new_col < BOARD_COLS):
                        break  # Out of bounds

                    if board[new_row, new_col] == player:
                        count += 1
                    else:
                        break  # Sequence broken

                if count == 4:
                    return True  # Found four in a row

    return False  # No four-in-a-row found


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Evaluate the state of the game from player's perspective after they have just moved.

    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player who just moved.

    Returns
    -------
    GameState
        The state of the game (win, draw, or still playing).
    """
    if connected_four(board, player):
        return GameState.IS_WIN

    if np.all(board != NO_PLAYER):
        return GameState.IS_DRAW

    return GameState.STILL_PLAYING


def _is_player_action(value: Any) -> bool:
    """
    Return True iff value is exactly of type numpy.int8. We avoid accepting Python int here so that tests catch accidental type changes early.

    Parameters
    ----------
    value : Any
        Value to check.

    Returns
    -------
    bool
        True if value is exactly numpy.int8, else False.
    """
    return type(value) is np.int8


def check_move_status(board: np.ndarray, column: Any) -> MoveStatus:
    """
    Validate whether a proposed column is a legal move on board.

    The function performs static validation only and does not mutate the board.

    Parameters
    ----------
    board : np.ndarray
        The game board.
    column : Any
        The column to check.

    Returns
    -------
    MoveStatus
        The status of the move (valid, wrong type, out of bounds, or full column).
    """
    if not _is_player_action(column):
        return MoveStatus.WRONG_TYPE  # Must be numpy.int8

    col_index = int(column)

    if not 0 <= col_index < BOARD_COLS:
        return MoveStatus.OUT_OF_BOUNDS  # Out of bounds

    if board[INDEX_HIGHEST_ROW, col_index] != NO_PLAYER:
        return MoveStatus.FULL_COLUMN  # Top cell is occupied

    return MoveStatus.IS_VALID
