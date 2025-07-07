"""
Main minimax agent implementation.

This module provides the main entry point for the minimax agent, combining
all the components (search, heuristics, transposition table, etc.) into
a cohesive agent.
"""

import time
from typing import Optional, Tuple

import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, PLAYER1, PLAYER2

from .saved_state import MinimaxSavedState
from .search_continuation import iterative_deepening_search, get_search_statistics


def generate_move_time_limited(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[SavedState] = None,
    time_limit_secs: float = 5.0,
    max_depth: int = 20
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generate a move for the given player using iterative deepening Negamax with alpha-beta pruning.
    
    This is the main entry point for the minimax agent. It uses iterative deepening
    to search as deeply as possible within the time limit, using a transposition table
    to store previously computed positions and sophisticated heuristics to evaluate
    board positions.

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
    max_depth : int, optional
        Maximum search depth (default is 20).

    Returns
    -------
    Tuple[PlayerAction, Optional[SavedState]]
        The chosen move and updated state.
        
    Raises
    ------
    ValueError
        If input parameters are invalid.
    """
    # Input validation
    if time_limit_secs <= 0:
        raise ValueError("Time limit must be positive")
    if max_depth <= 0:
        raise ValueError("Maximum depth must be positive")
    if player not in [PLAYER1, PLAYER2]:
        raise ValueError("Player must be PLAYER1 or PLAYER2")
    
    start_time = time.monotonic()
    
    # Initialize or reuse saved state
    if saved_state is None or not isinstance(saved_state, MinimaxSavedState):
        state = MinimaxSavedState()
    else:
        state = saved_state
    
    # Perform iterative deepening search
    try:
        best_col, best_score, completed_depth = iterative_deepening_search(
            board, player, state.transposition_table, time_limit_secs, max_depth
        )
    except ValueError as e:
        # Re-raise validation errors
        raise e
    
    # Calculate search time
    search_time = time.monotonic() - start_time
    
    # Store the best move for the root position
    if completed_depth > 0:
        state.transposition_table.store_position(
            board, completed_depth, best_score, PlayerAction(best_col), 'exact'
        )
    
    # Optional: Print search statistics for debugging
    if __debug__:
        stats = get_search_statistics(
            state.transposition_table, completed_depth, search_time
        )
        print(f"Search completed: depth {stats['completed_depth']}, "
              f"time {stats['search_time']:.3f}s, "
              f"TT size {stats['tt_size']}")
    
    return PlayerAction(best_col), state


def generate_move(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[SavedState] = None
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generate a move with default time limit.
    
    Convenience function that calls generate_move_time_limited with default parameters.
    
    Parameters
    ----------
    board : np.ndarray
        The current game board.
    player : BoardPiece
        The player to move.
    saved_state : Optional[SavedState], optional
        State to persist between moves.
        
    Returns
    -------
    Tuple[PlayerAction, Optional[SavedState]]
        The chosen move and updated state.
    """
    return generate_move_time_limited(board, player, saved_state)


def create_minimax_agent(
    time_limit: float = 5.0,
    max_depth: int = 20,
    max_table_size: int = 1000000
) -> tuple:
    """
    Create a configured minimax agent.
    
    Returns a tuple of (move_function, initial_state) that can be used
    with the game framework.
    
    Parameters
    ----------
    time_limit : float
        Time limit for move generation in seconds.
    max_depth : int
        Maximum search depth.
    max_table_size : int
        Maximum size of the transposition table.
        
    Returns
    -------
    tuple
        A tuple of (move_function, initial_state).
    """
    initial_state = MinimaxSavedState(max_table_size)
    
    def move_function(board: np.ndarray, player: BoardPiece, 
                     saved_state: Optional[SavedState] = None):
        return generate_move_time_limited(
            board, player, saved_state, time_limit, max_depth
        )
    
    return move_function, initial_state
