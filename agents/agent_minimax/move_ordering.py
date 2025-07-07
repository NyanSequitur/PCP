"""
Move ordering utilities for minimax agent.

This module provides functions to order moves for better alpha-beta pruning efficiency.
Good move ordering can significantly improve search performance by causing more cutoffs.
"""

import numpy as np
from typing import Dict, List

from game_utils import PlayerAction, BOARD_COLS, check_move_status, MoveStatus
from .transposition_table import TranspositionTable


def get_valid_columns(board: np.ndarray) -> List[int]:
    """
    Get all valid columns for move placement.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
        
    Returns
    -------
    List[int]
        List of valid column indices.
    """
    return [c for c in range(BOARD_COLS) 
            if check_move_status(board, PlayerAction(c)) == MoveStatus.IS_VALID]


def order_moves(board: np.ndarray, valid_cols: List[int], 
                tt: TranspositionTable) -> List[int]:
    """
    Order moves for better alpha-beta pruning efficiency.
    
    The ordering strategy is:
    1. Try the best move from the transposition table first
    2. Prioritize center columns (generally stronger in Connect Four)
    3. Order remaining columns by distance from center
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    valid_cols : List[int]
        List of valid column indices.
    tt : TranspositionTable
        The transposition table for move lookup.
        
    Returns
    -------
    List[int]
        Ordered list of column indices.
    """
    ordered_moves = []
    remaining_cols = valid_cols.copy()
    
    # Try transposition table best move first
    tt_best_move = tt.get_best_move(board)
    if tt_best_move is not None and int(tt_best_move) in remaining_cols:
        ordered_moves.append(int(tt_best_move))
        remaining_cols.remove(int(tt_best_move))
    
    # Add center columns first (better move ordering for Connect Four)
    center = BOARD_COLS // 2
    for offset in range(BOARD_COLS):
        if not remaining_cols:  # No more columns to add
            break
        
        # Try center column first
        if offset == 0:
            if center in remaining_cols:
                ordered_moves.append(center)
                remaining_cols.remove(center)
        else:
            # Try columns to the right and left of center
            for direction in [1, -1]:
                col = center + direction * offset
                if 0 <= col < BOARD_COLS and col in remaining_cols:
                    ordered_moves.append(col)
                    remaining_cols.remove(col)
                    break
    
    # Add any remaining columns (should not happen with correct logic)
    ordered_moves.extend(remaining_cols)
    return ordered_moves


class MoveOrderingCache:
    """
    Cache for storing move orderings for specific positions.
    
    This can be used to remember good move orderings for positions
    that are searched frequently.
    """
    
    def __init__(self):
        """Initialize the move ordering cache."""
        self.cache: Dict[str, List[PlayerAction]] = {}
    
    def get_move_ordering(self, board_hash: str) -> List[PlayerAction]:
        """
        Get move ordering for a position.
        
        Parameters
        ----------
        board_hash : str
            The hash of the board position.
            
        Returns
        -------
        List[PlayerAction]
            List of ordered moves, or empty list if not found.
        """
        return self.cache.get(board_hash, [])
    
    def store_move_ordering(self, board_hash: str, moves: List[PlayerAction]):
        """
        Store move ordering for a position.
        
        Parameters
        ----------
        board_hash : str
            The hash of the board position.
        moves : List[PlayerAction]
            The ordered list of moves.
        """
        self.cache[board_hash] = moves
