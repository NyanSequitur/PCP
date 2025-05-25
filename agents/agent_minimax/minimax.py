"""
Minimax agent for Connect Four with iterative deepening and alpha-beta pruning.

Implements a time-limited move generator using Negamax with alpha-beta pruning and a simple heuristic.
"""

import time
import math
import numpy as np
from typing import Optional, Tuple

from game_utils import (
    BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2,
    apply_player_action, check_end_state, GameState,
    check_move_status, MoveStatus, PlayerAction, BoardPiece, SavedState
)

def generate_move_time_limited(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[SavedState] = None,
    time_limit_secs: float = 5.0
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generate a move for the given player using iterative deepening Negamax with alpha-beta pruning.

    Parameters
    ----------
    board : np.ndarray
        The current game board.
    player : BoardPiece
        The player to move.
    saved_state : Optional[SavedState], optional
        State to persist between moves (default is None).
    time_limit_secs : float, optional
        Time limit for move search in seconds (default is 5.0).

    Returns
    -------
    Tuple[PlayerAction, Optional[SavedState]]
        The chosen move and updated state.
    """
    start_time = time.monotonic()
    deadline = start_time + time_limit_secs

    # Negamax with alpha-beta pruning: recursively evaluate moves for both players
    def _negamax(
        b: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        turn: BoardPiece
    ) -> float:
        """
        Negamax search with alpha-beta pruning.

        Parameters
        ----------
        b : np.ndarray
            Board state.
        depth : int
            Search depth.
        alpha : float
            Alpha value.
        beta : float
            Beta value.
        turn : BoardPiece
            Player to move.

        Returns
        -------
        float
            Heuristic value of the board.
        """
        if time.monotonic() > deadline:
            raise TimeoutError  # Stop search if time limit exceeded
        prev = _other(turn)
        state = check_end_state(b, prev)
        if state == GameState.IS_WIN:
            return -math.inf  # Previous player just won
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
            # Negamax recursion: switch player and invert score
            score = -_negamax(new_board, depth - 1, -beta, -alpha, _other(turn))
            value = max(value, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Beta cutoff
        return value

    valid_cols = [c for c in range(BOARD_COLS)
                  if check_move_status(board, PlayerAction(c)) == MoveStatus.IS_VALID]
    if not valid_cols:
        raise Exception("No valid moves available.")
    best_col = valid_cols[0]
    best_score = -math.inf
    depth = 1

    # Iterative deepening: increase search depth until time runs out
    try:
        while True:
            if time.monotonic() > deadline:
                break
            alpha = -math.inf
            beta = math.inf
            local_best_score = -math.inf
            local_best_col = valid_cols[0]
            # Try all valid columns at this depth
            for col in valid_cols:
                if time.monotonic() > deadline:
                    raise TimeoutError
                trial_board = board.copy()
                apply_player_action(trial_board, PlayerAction(col), player)
                # Negamax search for opponent's best response
                score = -_negamax(trial_board, depth - 1, -beta, -alpha, _other(player))
                if score > local_best_score:
                    local_best_score = score
                    local_best_col = col
                alpha = max(alpha, score)
            best_score = local_best_score
            best_col = local_best_col
            depth += 1
    except TimeoutError:
        pass  # Return the best move found so far if time runs out
    return PlayerAction(best_col), saved_state

def _other(player: BoardPiece) -> BoardPiece:
    """
    Return the opponent of the given player.

    Parameters
    ----------
    player : BoardPiece
        The current player.

    Returns
    -------
    BoardPiece
        The opponent player.
    """
    return PLAYER1 if player == PLAYER2 else PLAYER2

def _heuristic(board: np.ndarray, player: BoardPiece) -> float:
    """
    Heuristic evaluation of the board for the given player.
    Considers center column control and open-ended 2/3-in-a-row windows.

    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to evaluate for.

    Returns
    -------
    float
        Heuristic score.
    """
    def score_window(window: np.ndarray) -> float:
        cnt_player = np.count_nonzero(window == player)
        cnt_opponent = np.count_nonzero(window == _other(player))
        cnt_empty = np.count_nonzero(window == NO_PLAYER)
        if cnt_player == 4:
            return 100.0  # Win
        if cnt_player == 3 and cnt_empty == 1:
            return 5.0   # Three in a row
        if cnt_player == 2 and cnt_empty == 2:
            return 2.0   # Two in a row
        if cnt_opponent == 3 and cnt_empty == 1:
            return -4.0  # Block opponent
        return 0.0

    score = 0.0
    center_col = BOARD_COLS // 2
    center_count = int(np.count_nonzero(board[:, center_col] == player))
    score += center_count * 3.0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS - 3):
            score += score_window(board[r, c:c+4])
    for c in range(BOARD_COLS):
        for r in range(BOARD_ROWS - 3):
            score += score_window(board[r:r+4, c])
    for r in range(BOARD_ROWS - 3):
        for c in range(BOARD_COLS - 3):
            window = np.array([board[r+i, c+i] for i in range(4)])
            score += score_window(window)
    for r in range(BOARD_ROWS - 3):
        for c in range(3, BOARD_COLS):
            window = np.array([board[r+i, c-i] for i in range(4)])
            score += score_window(window)
    return score
