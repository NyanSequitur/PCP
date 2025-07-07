"""
Saved state management for minimax agent.

This module handles the persistent state that the minimax agent maintains
between moves, including the transposition table and move ordering cache.
"""

from typing import Optional
import numpy as np

from game_utils import SavedState, PlayerAction
from .transposition_table import TranspositionTable
from .move_ordering import MoveOrderingCache


class MinimaxSavedState(SavedState):
    """
    Saved state for minimax agent with transposition table and move ordering cache.
    
    This class maintains the persistent state that the minimax agent uses
    to improve performance across multiple moves:
    - Transposition table: Stores previously computed positions
    - Move ordering cache: Stores good move orderings for positions
    
    Attributes
    ----------
    transposition_table : TranspositionTable
        The transposition table for storing computed positions.
    move_ordering_cache : MoveOrderingCache
        Cache for storing move orderings.
    """
    
    def __init__(self, max_table_size: int = 1000000):
        """
        Initialize the saved state.
        
        Parameters
        ----------
        max_table_size : int
            Maximum size of the transposition table.
        """
        self.transposition_table = TranspositionTable(max_table_size)
        self.move_ordering_cache = MoveOrderingCache()
    
    def get_board_hash(self, board: np.ndarray) -> str:
        """
        Get a hash string for the board state.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        str
            Hash string of the board.
        """
        return self.transposition_table.get_board_hash(board)
    
    def get_best_move(self, board: np.ndarray) -> Optional[PlayerAction]:
        """
        Get the best move for a position if available.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        Optional[PlayerAction]
            The best move if available, None otherwise.
        """
        return self.transposition_table.get_best_move(board)
    
    def store_position(self, board: np.ndarray, depth: int, value: float, 
                      best_move: Optional[PlayerAction], flag: str):
        """
        Store a position in the transposition table.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
        depth : int
            The depth at which this position was evaluated.
        value : float
            The evaluated value of the position.
        best_move : Optional[PlayerAction]
            The best move found for this position.
        flag : str
            The type of bound: 'exact', 'lower', or 'upper'.
        """
        self.transposition_table.store_position(board, depth, value, best_move, flag)
    
    def lookup_position(self, board: np.ndarray, depth: int, alpha: float, beta: float):
        """
        Look up a position in the transposition table.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
        depth : int
            The required minimum depth.
        alpha : float
            The alpha bound.
        beta : float
            The beta bound.
            
        Returns
        -------
        tuple[bool, float]
            A tuple of (found, value) where found indicates if a usable
            entry was found and value is the stored value.
        """
        return self.transposition_table.lookup_position(board, depth, alpha, beta)
    
    def get_move_ordering(self, board: np.ndarray) -> list[PlayerAction]:
        """
        Get move ordering for a position.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        list[PlayerAction]
            List of ordered moves.
        """
        board_hash = self.get_board_hash(board)
        return self.move_ordering_cache.get_move_ordering(board_hash)
    
    def store_move_ordering(self, board: np.ndarray, moves: list[PlayerAction]):
        """
        Store move ordering for a position.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
        moves : list[PlayerAction]
            The ordered list of moves.
        """
        board_hash = self.get_board_hash(board)
        self.move_ordering_cache.store_move_ordering(board_hash, moves)
    
    def clear_cache(self):
        """
        Clear all cached data.
        
        This can be useful to free memory or when starting a new game
        where previous positions are no longer relevant.
        """
        self.transposition_table.table.clear()
        self.move_ordering_cache.cache.clear()
    
    def get_cache_statistics(self) -> dict:
        """
        Get statistics about the cached data.
        
        Returns
        -------
        dict
            Dictionary containing cache statistics.
        """
        return {
            'tt_entries': len(self.transposition_table.table),
            'tt_max_size': self.transposition_table.max_table_size,
            'move_ordering_entries': len(self.move_ordering_cache.cache),
            'tt_utilization': len(self.transposition_table.table) / self.transposition_table.max_table_size
        }
