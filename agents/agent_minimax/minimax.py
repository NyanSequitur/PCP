# minimax.py

import time
import math
import numpy as np
from typing import Optional

from game_utils import (
    BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2,
    apply_player_action, check_end_state, GameState,
    check_move_status, MoveStatus, PlayerAction, BoardPiece, SavedState
)

def generate_move_time_limited(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[SavedState] = None,
    time_limit_secs: float = 10.0
) -> tuple[PlayerAction, Optional[SavedState]]:
    """
    Choose the best move found within the given time limit (in seconds) using iterative deepening
    with Negamax and alpha-beta pruning.
    """
    start_time = time.monotonic()
    deadline = start_time + time_limit_secs

    # helper negamax defined inside to access deadline
    def _negamax_inner(
        b: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        turn: BoardPiece
    ) -> float:
        # time check
        if time.monotonic() > deadline:
            raise TimeoutError

        # Check terminal on previous move
        prev = _other(turn)
        state = check_end_state(b, prev)
        if state == GameState.IS_WIN:
            return -math.inf
        if state == GameState.IS_DRAW:
            return 0.0

        if depth == 0:
            return _heuristic(b, turn)

        value = -math.inf
        for c in range(BOARD_COLS):
            if check_move_status(b, PlayerAction(c)) != MoveStatus.IS_VALID:
                continue
            new_board = b.copy()
            apply_player_action(new_board, PlayerAction(c), turn)
            score = -_negamax_inner(new_board, depth - 1, -beta, -alpha, _other(turn))
            value = max(value, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return value

    # valid moves
    valid_cols = [c for c in range(BOARD_COLS)
                  if check_move_status(board, PlayerAction(c)) == MoveStatus.IS_VALID]
    best_col = valid_cols[0]
    best_score = -math.inf
    depth = 1

    try:
        # iterative deepening
        while True:
            # time check before each depth iteration
            if time.monotonic() > deadline:
                break

            alpha = -math.inf
            beta = math.inf
            local_best_score = -math.inf
            local_best_col = valid_cols[0]

            for col in valid_cols:
                if time.monotonic() > deadline:
                    raise TimeoutError
                trial_board = board.copy()
                apply_player_action(trial_board, PlayerAction(col), player)
                score = -_negamax_inner(trial_board, depth - 1, -beta, -alpha, _other(player))
                if score > local_best_score:
                    local_best_score = score
                    local_best_col = col
                alpha = max(alpha, score)

            # update best move if this depth completed
            best_score = local_best_score
            best_col = local_best_col
            depth += 1
            print(f"Depth {depth} completed, best score: {best_score:.2f}, move: {best_col}")
    except TimeoutError:
        # time expired, return the best move from last completed depth
        pass
    return PlayerAction(best_col), saved_state


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
            score += score_window(board[r, c:c+4])

    # vertical
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS - 3):
            score += score_window(board[r:r+4, c])

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
