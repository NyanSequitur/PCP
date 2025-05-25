"""
Random agent for Connect Four.

Selects a valid move at random from available columns.
"""

import numpy as np
import random

from game_utils import BoardPiece, PlayerAction, SavedState, MoveStatus, check_move_status, BOARD_COLS


def generate_move_random(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    """
    Select a random valid move for the player.

    Parameters
    ----------
    board : np.ndarray
        The current game board.
    player : BoardPiece
        The player to move.
    saved_state : SavedState | None
        State to persist between moves.

    Returns
    -------
    tuple[PlayerAction, SavedState | None]
        The chosen move and updated state.
    """
    # Collect all valid, non-full columns for random selection
    valid_cols = [
        c for c in range(BOARD_COLS)
        if check_move_status(board, PlayerAction(c)) == MoveStatus.IS_VALID
    ]
    # Randomly choose one valid column
    choice = random.choice(valid_cols)
    return PlayerAction(choice), saved_state