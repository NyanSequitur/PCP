"""
Core search algorithms for minimax agent.

This module contains the negamax search implementation with alpha-beta pruning
and the search continuation logic for iterative deepening.
"""

import time
import math
import numpy as np
from typing import Tuple, List

from game_utils import (
    BoardPiece, PlayerAction, apply_player_action, check_end_state, GameState
)
from .transposition_table import TranspositionTable
from .move_ordering import get_valid_columns, order_moves
from .heuristics import evaluate_board, get_other_player


def get_transposition_flag(value: float, original_alpha: float, beta: float) -> str:
    """
    Determine the transposition table flag based on alpha-beta bounds.
    
    Parameters
    ----------
    value : float
        The evaluated value.
    original_alpha : float
        The original alpha value before search.
    beta : float
        The beta value.
        
    Returns
    -------
    str
        The flag: 'exact', 'upper', or 'lower'.
    """
    if value <= original_alpha:
        return 'upper'
    elif value >= beta:
        return 'lower'
    else:
        return 'exact'


def negamax_search(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    turn: BoardPiece,
    tt: TranspositionTable,
    deadline: float
) -> float:
    """
    Negamax search with alpha-beta pruning and transposition table.
    
    Negamax is a variant of minimax that simplifies the implementation by
    using the zero-sum property of the game. Instead of separate min and max
    functions, it uses a single function that negates the result.

    Parameters
    ----------
    board : np.ndarray
        Board state.
    depth : int
        Search depth.
    alpha : float
        Alpha value (best value the maximizing player can guarantee).
    beta : float
        Beta value (best value the minimizing player can guarantee).
    turn : BoardPiece
        Player to move.
    tt : TranspositionTable
        The transposition table for position lookup and storage.
    deadline : float
        Time deadline for search.

    Returns
    -------
    float
        Heuristic value of the board from the current player's perspective.
        
    Raises
    ------
    TimeoutError
        If the search exceeds the time deadline.
    """
    if time.monotonic() > deadline:
        raise TimeoutError("Search time limit exceeded")
    
    # Check transposition table first
    found, tt_value = tt.lookup_position(board, depth, alpha, beta)
    if found:
        return tt_value
    
    original_alpha = alpha
    
    # Check for terminal states
    previous_player = get_other_player(turn)
    game_state = check_end_state(board, previous_player)
    if game_state == GameState.IS_WIN:
        return -math.inf  # Previous player just won
    if game_state == GameState.IS_DRAW:
        return 0.0
    if depth == 0:
        return evaluate_board(board, turn)
    
    # Search all moves
    value = -math.inf
    best_move = None
    valid_cols = get_valid_columns(board)
    
    # Handle case where no valid moves exist (should not happen in normal game)
    if not valid_cols:
        return 0.0  # Return neutral score if no moves available
    
    ordered_moves = order_moves(board, valid_cols, tt)
    
    for col in ordered_moves:
        if time.monotonic() > deadline:
            raise TimeoutError("Search time limit exceeded")
        
        new_board = board.copy()
        apply_player_action(new_board, PlayerAction(col), turn)
        
        # Negamax recursion: switch player and invert score
        score = -negamax_search(new_board, depth - 1, -beta, -alpha, 
                               get_other_player(turn), tt, deadline)
        
        if score > value:
            value = score
            best_move = PlayerAction(col)
        
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # Beta cutoff - opponent won't allow this path
    
    # Store in transposition table
    flag = get_transposition_flag(value, original_alpha, beta)
    tt.store_position(board, depth, value, best_move, flag)
    
    return value


def search_at_depth(
    board: np.ndarray,
    player: BoardPiece,
    depth: int,
    valid_cols: List[int],
    tt: TranspositionTable,
    deadline: float
) -> Tuple[int, float]:
    """
    Search at a specific depth and return the best move and score.
    
    This function performs a complete search at the root level, evaluating
    all valid moves and returning the best one found.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to move.
    depth : int
        Search depth.
    valid_cols : List[int]
        List of valid column indices.
    tt : TranspositionTable
        The transposition table for position lookup and storage.
    deadline : float
        Time deadline for search.
        
    Returns
    -------
    Tuple[int, float]
        Best column and its score.
        
    Raises
    ------
    TimeoutError
        If the search exceeds the time deadline.
    """
    alpha = -math.inf
    beta = math.inf
    best_score = -math.inf
    best_col = valid_cols[0]
    
    # Order moves for this depth
    ordered_cols = order_moves(board, valid_cols, tt)
    
    for col in ordered_cols:
        if time.monotonic() > deadline:
            raise TimeoutError("Search time limit exceeded")
        
        trial_board = board.copy()
        apply_player_action(trial_board, PlayerAction(col), player)
        
        # Negamax search for opponent's best response
        score = -negamax_search(trial_board, depth - 1, -beta, -alpha, 
                               get_other_player(player), tt, deadline)
        
        if score > best_score:
            best_score = score
            best_col = col
        
        alpha = max(alpha, score)
    
    return best_col, best_score
