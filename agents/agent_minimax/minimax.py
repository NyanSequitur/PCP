import math
import numpy as np
from typing import Optional

from game_utils import (
    BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2,
    apply_player_action, check_end_state, GameState,
    check_move_status, MoveStatus, PlayerAction, BoardPiece, SavedState
)

def generate_move_minimax(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: SavedState | None,
    depth: int = 4
) -> tuple[PlayerAction, SavedState | None]:
    """
    Choose the column with the highest Negamax score (with alpha-beta pruning).
    depth: number of plies to look ahead.
    """
    valid_cols = [
        c for c in range(BOARD_COLS)
        if check_move_status(board, PlayerAction(c)) == MoveStatus.IS_VALID
    ]
    best_score = -math.inf
    best_col = valid_cols[0]
    alpha = -math.inf
    beta = math.inf

    for col in valid_cols:
        # simulate
        temp = board.copy()
        apply_player_action(temp, PlayerAction(col), player)
        # opponent to move next, so flip sign
        score = -_negamax(temp, depth - 1, -beta, -alpha, _other(player))
        if score > best_score:
            best_score = score
            best_col = col
        alpha = max(alpha, score)

    return PlayerAction(best_col), saved_state


def _negamax(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    player: BoardPiece
) -> float:
    """
    Internal negamax recursive helper.
    Returns the best score from `player`'s perspective.
    """
    # Check terminal on **previous** move, i.e. other player
    prev = _other(player)
    state = check_end_state(board, prev)
    if state == GameState.IS_WIN:
        # previous player just won
        return -math.inf
    if state == GameState.IS_DRAW:
        return 0.0

    if depth == 0:
        return _heuristic(board, player)

    value = -math.inf
    for col in range(BOARD_COLS):
        if check_move_status(board, PlayerAction(col)) != MoveStatus.IS_VALID:
            continue
        temp = board.copy()
        apply_player_action(temp, PlayerAction(col), player)
        score = -_negamax(temp, depth - 1, -beta, -alpha, _other(player))
        value = max(value, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # alpha-beta cutoff

    return value


def _other(player: BoardPiece) -> BoardPiece:
    return PLAYER1 if player == PLAYER2 else PLAYER2


def _heuristic(board: np.ndarray, player: BoardPiece) -> float:
    """
    A simple evaluation: 
      - center‐column control,
      - count how many 2/3 in‐a‐row windows with open ends.
    Positive = good for `player`, negative = good for opponent.
    """
    def score_window(window):
        cnt_player = np.count_nonzero(window == player)
        cnt_opponent = np.count_nonzero(window == _other(player))
        cnt_empty = np.count_nonzero(window == NO_PLAYER)
        if cnt_player == 4:
            return 100.0
        if cnt_player == 3 and cnt_empty == 1:
            return 5.0
        if cnt_player == 2 and cnt_empty == 2:
            return 2.0
        if cnt_opponent == 3 and cnt_empty == 1:
            return -4.0
        return 0.0

    score = 0.0
    # center column preference
    center_col = BOARD_COLS // 2
    center_count = int(np.count_nonzero(board[:, center_col] == player))
    score += center_count * 3.0

    # all windows of length 4 horizontally
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            window = board[r, c:c+4]
            score += score_window(window)

    # vertical
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS - 3):
            window = board[r:r+4, c]
            score += score_window(window)

    # diagonal up-right
    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS - 3):
            window = np.array([board[r+i, c+i] for i in range(4)])
            score += score_window(window)

    # diagonal up-left
    for r in range(BOARD_ROWS - 3):
        for c in range(3, BOARD_COLS):
            window = np.array([board[r+i, c-i] for i in range(4)])
            score += score_window(window)

    return score
