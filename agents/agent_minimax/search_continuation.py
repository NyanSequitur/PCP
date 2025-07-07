"""
Search continuation and iterative deepening for minimax agent.

This module handles the iterative deepening search logic, which gradually
increases the search depth until time runs out.
"""

import time
from typing import Tuple, List

import numpy as np
from game_utils import BoardPiece, PlayerAction

from .transposition_table import TranspositionTable
from .move_ordering import get_valid_columns
from .search import search_at_depth


def iterative_deepening_search(
    board: np.ndarray,
    player: BoardPiece,
    tt: TranspositionTable,
    time_limit_secs: float,
    max_depth: int
) -> Tuple[int, float, int]:
    """
    Perform iterative deepening search to find the best move.
    
    Iterative deepening gradually increases the search depth until time runs out.
    This approach has several advantages:
    1. It provides a good move quickly (from shallow search)
    2. It can be interrupted at any time with a reasonable result
    3. The overhead of re-searching shallow depths is minimal due to exponential
       growth of search tree size
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to move.
    tt : TranspositionTable
        The transposition table for position lookup and storage.
    time_limit_secs : float
        Time limit for search in seconds.
    max_depth : int
        Maximum search depth.
        
    Returns
    -------
    Tuple[int, float, int]
        A tuple of (best_column, best_score, completed_depth).
        
    Raises
    ------
    ValueError
        If no valid moves are available.
    """
    start_time = time.monotonic()
    deadline = start_time + time_limit_secs
    
    valid_cols = get_valid_columns(board)
    if not valid_cols:
        raise ValueError("No valid moves available")
    
    # Initialize with first valid move
    best_col = valid_cols[0]
    best_score = -float('inf')
    completed_depth = 0
    
    # Check if we have a best move from transposition table
    tt_best_move = tt.get_best_move(board)
    if tt_best_move is not None and int(tt_best_move) in valid_cols:
        best_col = int(tt_best_move)
    
    # Iterative deepening: increase search depth until time runs out
    depth = 1
    try:
        while depth <= max_depth and time.monotonic() < deadline:
            # Search at current depth
            local_best_col, local_best_score = search_at_depth(
                board, player, depth, valid_cols, tt, deadline
            )
            
            # Update best move if search completed successfully
            best_score = local_best_score
            best_col = local_best_col
            completed_depth = depth
            
            # Check if we found a winning move
            if best_score >= 1000.0:  # Win threshold
                break
            
            depth += 1
            
    except TimeoutError:
        pass  # Return the best move found so far
    
    return best_col, best_score, completed_depth


def should_extend_search(
    board: np.ndarray,
    player: BoardPiece,
    current_depth: int,
    best_score: float,
    time_remaining: float
) -> bool:
    """
    Determine if the search should be extended based on various factors.
    
    This function can be used to implement selective search extensions,
    such as extending search when the position is critical or when there
    are forced moves.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to move.
    current_depth : int
        Current search depth.
    best_score : float
        Best score found so far.
    time_remaining : float
        Time remaining in seconds.
        
    Returns
    -------
    bool
        True if search should be extended.
    """
    # Don't extend if we're running out of time
    if time_remaining < 0.1:
        return False
    
    # Extend search if we found a winning move (to verify it's real)
    if abs(best_score) >= 1000.0:
        return current_depth < 10  # Reasonable upper bound
    
    # Extend search if position looks critical (many threats)
    # This could be enhanced with more sophisticated position analysis
    
    return False


def get_search_statistics(
    tt: TranspositionTable,
    completed_depth: int,
    search_time: float
) -> dict:
    """
    Get statistics about the search performance.
    
    Parameters
    ----------
    tt : TranspositionTable
        The transposition table used in search.
    completed_depth : int
        The maximum depth that was completed.
    search_time : float
        Time taken for the search.
        
    Returns
    -------
    dict
        Dictionary containing search statistics.
    """
    return {
        'completed_depth': completed_depth,
        'search_time': search_time,
        'tt_size': len(tt.table),
        'tt_max_size': tt.max_table_size,
        'tt_hit_rate': _estimate_tt_hit_rate(tt)
    }


def _estimate_tt_hit_rate(tt: TranspositionTable) -> float:
    """
    Estimate the hit rate of the transposition table.
    
    This is a simplified estimation. A more accurate implementation
    would track hits and misses during search.
    
    Parameters
    ----------
    tt : TranspositionTable
        The transposition table.
        
    Returns
    -------
    float
        Estimated hit rate as a percentage.
    """
    # Simple heuristic: fuller table suggests more hits
    fill_ratio = len(tt.table) / max(tt.max_table_size, 1)
    return min(fill_ratio * 100, 90.0)  # Cap at 90% as it's just an estimate
