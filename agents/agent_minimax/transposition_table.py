"""
Transposition table implementation for minimax agent.

This module handles the storage and retrieval of previously computed game positions
to avoid redundant calculations during search.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from game_utils import PlayerAction, BoardPiece, BOARD_COLS


@dataclass
class TranspositionEntry:
    """
    Entry in the transposition table.
    
    Attributes
    ----------
    value : float
        The evaluated value of the position.
    depth : int
        The depth at which this position was evaluated.
    best_move : Optional[PlayerAction]
        The best move found for this position.
    flag : str
        The type of bound: 'exact', 'lower' (beta cutoff), or 'upper' (alpha cutoff).
    """
    value: float
    depth: int
    best_move: Optional[PlayerAction]
    flag: str


class TranspositionTable:
    """
    Transposition table for storing previously computed positions.
    
    The table uses board symmetry to reduce memory usage by storing only
    the canonical form of each position.
    """
    
    def __init__(self, max_table_size: int = 1000000):
        """
        Initialize the transposition table.
        
        Parameters
        ----------
        max_table_size : int
            Maximum number of entries in the table before cleanup.
        """
        self.table: Dict[str, TranspositionEntry] = {}
        self.max_table_size = max_table_size
        
        # Statistics tracking
        self.lookups = 0
        self.hits = 0
        self.stores = 0
        self.collisions = 0
        self.nodes_searched = 0  # Track nodes for performance metrics
    
    def __len__(self) -> int:
        """Return the number of entries in the table."""
        return len(self.table)
    
    def _cleanup_table_if_needed(self):
        """
        Clean up transposition table if it gets too large.
        
        Uses a simple strategy of removing the oldest entries.
        A more sophisticated implementation could use LRU or depth-based cleanup.
        """
        if len(self.table) > self.max_table_size:
            items_to_remove = len(self.table) - self.max_table_size // 2
            keys_to_remove = list(self.table.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.table[key]
    
    def get_board_hash(self, board: np.ndarray) -> str:
        """
        Get a hash string for the board state using canonical form for symmetry.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        str
            Hexadecimal hash string of the canonical board.
        """
        canonical_board = self._get_canonical_board(board)
        return canonical_board.tobytes().hex()
    
    def _get_canonical_board(self, board: np.ndarray) -> np.ndarray:
        """
        Get the canonical form of the board by choosing the lexicographically smaller
        representation between the original and its horizontal mirror.
        
        This takes advantage of Connect Four's horizontal symmetry to reduce
        the number of unique positions we need to store.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        np.ndarray
            The canonical board representation.
        """
        # Mirror the board horizontally (flip left-right)
        mirrored = np.fliplr(board)
        
        # Compare boards lexicographically to determine canonical form
        original_flat = board.flatten()
        mirrored_flat = mirrored.flatten()
        
        # Return the lexicographically smaller one
        for i in range(len(original_flat)):
            if original_flat[i] < mirrored_flat[i]:
                return board
            elif original_flat[i] > mirrored_flat[i]:
                return mirrored
        
        # If they're equal, return the original
        return board
    
    def _is_board_mirrored(self, board: np.ndarray) -> bool:
        """
        Check if the canonical form is the mirrored version of the original board.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        bool
            True if the canonical form is the mirrored version.
        """
        mirrored = np.fliplr(board)
        original_flat = board.flatten()
        mirrored_flat = mirrored.flatten()
        
        # Check if mirrored version is lexicographically smaller
        for i in range(len(original_flat)):
            if original_flat[i] < mirrored_flat[i]:
                return False  # Original is smaller, not mirrored
            elif original_flat[i] > mirrored_flat[i]:
                return True   # Mirrored is smaller, so canonical is mirrored
        
        # If they're equal, canonical is original (not mirrored)
        return False
    
    def _mirror_move(self, move: PlayerAction) -> PlayerAction:
        """
        Mirror a move horizontally.
        
        Parameters
        ----------
        move : PlayerAction
            The original move.
            
        Returns
        -------
        PlayerAction
            The mirrored move.
        """
        return PlayerAction(BOARD_COLS - 1 - int(move))
    
    def store_position(self, board: np.ndarray, depth: int, value: float, 
                      best_move: Optional[PlayerAction], flag: str):
        """
        Store a position in the transposition table, handling symmetry.
        
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
        board_hash = self.get_board_hash(board)
        
        # If the canonical form is mirrored, mirror the best move before storing
        if best_move is not None and self._is_board_mirrored(board):
            best_move = self._mirror_move(best_move)
        
        # Check for collisions (existing entry with different value)
        if board_hash in self.table:
            existing = self.table[board_hash]
            if existing.depth < depth:  # Replace if this is deeper
                self.collisions += 1
        
        self.table[board_hash] = TranspositionEntry(
            value=value, depth=depth, best_move=best_move, flag=flag
        )
        self.stores += 1
        
        # Clean up table if it gets too large
        self._cleanup_table_if_needed()
    
    def lookup_position(self, board: np.ndarray, depth: int, alpha: float, beta: float):
        """
        Look up a position in the transposition table, handling symmetry.
        
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
        board_hash = self.get_board_hash(board)
        self.lookups += 1
        
        if board_hash not in self.table:
            return False, 0.0
        
        entry = self.table[board_hash]
        
        # Only use entries with sufficient depth
        if entry.depth < depth:
            return False, 0.0
        
        # We found a usable entry
        self.hits += 1
        
        # Check bounds and return appropriate values
        if entry.flag == 'exact':
            return True, entry.value
        elif entry.flag == 'lower' and entry.value >= beta:
            return True, entry.value
        elif entry.flag == 'upper' and entry.value <= alpha:
            return True, entry.value
        
        # Entry found but not usable due to bounds
        self.hits -= 1  # Don't count as hit if not usable
        return False, 0.0
    
    def get_stored_result(self, board: np.ndarray) -> Tuple[int, Optional[float], Optional[PlayerAction]]:
        """
        Get the stored depth, value, and best move for a position if available.
        
        This is useful for search continuation - if we have a good result stored,
        we can use it as our starting point for deeper search.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        Tuple[int, Optional[float], Optional[PlayerAction]]
            A tuple of (depth, value, best_move) where depth is the stored depth
            (0 if not found), value is the stored value (None if not found or not exact),
            and best_move is the best move (None if not found).
        """
        board_hash = self.get_board_hash(board)
        self.lookups += 1  # Track lookup for statistics
        
        if board_hash in self.table:
            entry = self.table[board_hash]
            best_move = entry.best_move
            
            # If the canonical form is mirrored, mirror the move back
            if best_move is not None and self._is_board_mirrored(board):
                best_move = self._mirror_move(best_move)
            
            # Only return value if it's exact (not a bound)
            value = entry.value if entry.flag == 'exact' else None
            
            self.hits += 1  # Count as hit since we found the entry
            return entry.depth, value, best_move
        
        return 0, None, None

    def get_stored_depth(self, board: np.ndarray) -> int:
        """
        Get the depth at which a position was previously stored in the table.
        
        This is useful for search continuation - if we've already analyzed
        a position deeply, we can start the next search from that depth
        rather than starting from depth 1.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        int
            The depth at which this position was stored, or 0 if not found.
        """
        board_hash = self.get_board_hash(board)
        if board_hash in self.table:
            return self.table[board_hash].depth
        return 0

    def get_best_move(self, board: np.ndarray) -> Optional[PlayerAction]:
        """
        Get the best move for a position if available, handling symmetry.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        Optional[PlayerAction]
            The best move if available, None otherwise.
        """
        board_hash = self.get_board_hash(board)
        if board_hash in self.table:
            best_move = self.table[board_hash].best_move
            if best_move is not None:
                # If the canonical form is mirrored, mirror the move back
                if self._is_board_mirrored(board):
                    return self._mirror_move(best_move)
                else:
                    return best_move
        return None

    def reset_statistics(self):
        """Reset all statistics counters."""
        self.lookups = 0
        self.hits = 0
        self.stores = 0
        self.collisions = 0
        self.nodes_searched = 0
    
    def get_hit_rate(self) -> float:
        """Get the current hit rate as a percentage."""
        return (self.hits / self.lookups * 100) if self.lookups > 0 else 0.0
