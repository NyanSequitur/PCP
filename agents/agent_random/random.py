import numpy as np
import random

from game_utils import BoardPiece, PlayerAction, SavedState, MoveStatus, check_move_status, BOARD_COLS


def generate_move_random(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    # collect all valid, non-full columns
    valid_cols = [
        c for c in range(BOARD_COLS)
        if check_move_status(board, PlayerAction(c)) == MoveStatus.IS_VALID
    ]
    # choose one at random
    choice = random.choice(valid_cols)
    return PlayerAction(choice), saved_state