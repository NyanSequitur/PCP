"""
Unit tests for the minimax transposition table functionality.

This module tests the transposition table implementation, including position storage,
retrieval, hash functions, and symmetry optimizations.
"""

import numpy as np
import pytest
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2, 
    BOARD_COLS, NO_PLAYER
)
from agents.agent_minimax.transposition_table import TranspositionTable, TranspositionEntry
from agents.agent_minimax.saved_state import MinimaxSavedState


class TestTranspositionTableBasic:
    """Test basic transposition table functionality."""
    
    def test_transposition_table_creation(self):
        """Test creating a transposition table."""
        tt = TranspositionTable()
        assert len(tt) == 0
        assert tt.max_table_size > 0
    
    def test_transposition_table_custom_size(self):
        """Test creating a transposition table with custom size."""
        tt = TranspositionTable(max_table_size=1000)
        assert tt.max_table_size == 1000
        assert len(tt) == 0
    
    def test_transposition_entry_creation(self):
        """Test creating transposition entries."""
        entry = TranspositionEntry(
            value=10.5,
            depth=5,
            best_move=PlayerAction(3),
            flag='exact'
        )
        
        assert entry.value == 10.5
        assert entry.depth == 5
        assert entry.best_move == PlayerAction(3)
        assert entry.flag == 'exact'
    
    def test_store_and_lookup_exact(self):
        """Test storing and looking up exact entries."""
        tt = TranspositionTable()
        board = initialize_game_state()
        
        # Store a position
        tt.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
        
        # Look it up
        found, value = tt.lookup_position(board, 5, -1000, 1000)
        assert found
        assert value == 10.0
    
    def test_store_and_lookup_lower_bound(self):
        """Test storing and looking up lower bound entries."""
        tt = TranspositionTable()
        board = initialize_game_state()
        
        # Store a lower bound position
        tt.store_position(board, 5, 10.0, PlayerAction(3), 'lower')
        
        # Look it up with beta less than or equal to stored value
        found, value = tt.lookup_position(board, 5, 5.0, 10.0)
        assert found
        assert value == 10.0
        
        # Look it up with beta greater than stored value
        found, value = tt.lookup_position(board, 5, 5.0, 15.0)
        assert not found
    
    def test_store_and_lookup_upper_bound(self):
        """Test storing and looking up upper bound entries."""
        tt = TranspositionTable()
        board = initialize_game_state()
        
        # Store an upper bound position
        tt.store_position(board, 5, 10.0, PlayerAction(3), 'upper')
        
        # Look it up with alpha greater than or equal to stored value
        found, value = tt.lookup_position(board, 5, 10.0, 15.0)
        assert found
        assert value == 10.0
        
        # Look it up with alpha less than stored value
        found, value = tt.lookup_position(board, 5, 5.0, 15.0)
        assert not found
    
    def test_lookup_nonexistent_position(self):
        """Test looking up a position that doesn't exist."""
        tt = TranspositionTable()
        board = initialize_game_state()
        
        found, value = tt.lookup_position(board, 5, -1000, 1000)
        assert not found
        assert value == 0.0
    
    def test_lookup_insufficient_depth(self):
        """Test looking up position with insufficient depth."""
        tt = TranspositionTable()
        board = initialize_game_state()
        
        # Store at depth 5
        tt.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
        
        # Look up at depth 6 (should not find)
        found, value = tt.lookup_position(board, 6, -1000, 1000)
        assert not found
        
        # Look up at depth 5 (should find)
        found, value = tt.lookup_position(board, 5, -1000, 1000)
        assert found
        assert value == 10.0
        
        # Look up at depth 4 (should find)
        found, value = tt.lookup_position(board, 4, -1000, 1000)
        assert found
        assert value == 10.0


class TestTranspositionTableBestMove:
    """Test best move storage and retrieval."""
    
    def test_store_and_get_best_move(self):
        """Test storing and retrieving best moves."""
        tt = TranspositionTable()
        board = initialize_game_state()
        
        # Store a position with best move
        tt.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
        
        # Get the best move
        best_move = tt.get_best_move(board)
        assert best_move == PlayerAction(3)
    
    def test_get_best_move_nonexistent(self):
        """Test getting best move for nonexistent position."""
        tt = TranspositionTable()
        board = initialize_game_state()
        
        best_move = tt.get_best_move(board)
        assert best_move is None
    
    def test_update_best_move(self):
        """Test updating best move for existing position."""
        tt = TranspositionTable()
        board = initialize_game_state()
        
        # Store initial position
        tt.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
        
        # Update with better information
        tt.store_position(board, 6, 12.0, PlayerAction(4), 'exact')
        
        # Should have updated best move
        best_move = tt.get_best_move(board)
        assert best_move == PlayerAction(4)


class TestTranspositionTableHashing:
    """Test board hashing functionality."""
    
    def test_same_boards_same_hash(self):
        """Test that identical boards have the same hash."""
        tt = TranspositionTable()
        board1 = initialize_game_state()
        board2 = initialize_game_state()
        
        hash1 = tt.get_board_hash(board1)
        hash2 = tt.get_board_hash(board2)
        
        assert hash1 == hash2
    
    def test_different_boards_different_hash(self):
        """Test that different boards have different hashes."""
        tt = TranspositionTable()
        board1 = initialize_game_state()
        board2 = initialize_game_state()
        
        # Make boards different
        apply_player_action(board2, PlayerAction(3), PLAYER1)
        
        hash1 = tt.get_board_hash(board1)
        hash2 = tt.get_board_hash(board2)
        
        assert hash1 != hash2
    
    def test_board_hash_consistency(self):
        """Test that board hashes are consistent."""
        tt = TranspositionTable()
        board = initialize_game_state()
        apply_player_action(board, PlayerAction(3), PLAYER1)
        
        # Hash should be the same when computed multiple times
        hash1 = tt.get_board_hash(board)
        hash2 = tt.get_board_hash(board)
        
        assert hash1 == hash2


class TestTranspositionTableSymmetry:
    """Test symmetry optimization in transposition table."""
    
    def test_mirror_boards_same_hash(self):
        """Test that mirrored boards have the same hash."""
        tt = TranspositionTable()
        
        # Create a board and its mirror
        board1 = initialize_game_state()
        apply_player_action(board1, PlayerAction(1), PLAYER1)
        apply_player_action(board1, PlayerAction(2), PLAYER2)
        
        board2 = initialize_game_state()
        apply_player_action(board2, PlayerAction(5), PLAYER1)  # Mirror of column 1
        apply_player_action(board2, PlayerAction(4), PLAYER2)  # Mirror of column 2
        
        hash1 = tt.get_board_hash(board1)
        hash2 = tt.get_board_hash(board2)
        
        assert hash1 == hash2
    
    def test_symmetric_position_storage(self):
        """Test that symmetric positions are stored correctly."""
        tt = TranspositionTable()
        
        # Create a board
        board = initialize_game_state()
        apply_player_action(board, PlayerAction(1), PLAYER1)
        
        # Store position
        tt.store_position(board, 5, 10.0, PlayerAction(1), 'exact')
        
        # Create mirrored board
        mirrored_board = initialize_game_state()
        apply_player_action(mirrored_board, PlayerAction(5), PLAYER1)
        
        # Should find the mirrored position
        found, value = tt.lookup_position(mirrored_board, 5, -1000, 1000)
        assert found
        assert value == 10.0
    
    def test_symmetric_best_move_mirroring(self):
        """Test that best moves are properly mirrored."""
        tt = TranspositionTable()
        
        # Create a board
        board = initialize_game_state()
        apply_player_action(board, PlayerAction(1), PLAYER1)
        
        # Store position with best move
        tt.store_position(board, 5, 10.0, PlayerAction(2), 'exact')
        
        # Create mirrored board
        mirrored_board = initialize_game_state()
        apply_player_action(mirrored_board, PlayerAction(5), PLAYER1)
        
        # Best move should be mirrored
        best_move = tt.get_best_move(mirrored_board)
        assert best_move == PlayerAction(4)  # Mirror of column 2


class TestTranspositionTableSize:
    """Test transposition table size management."""
    
    def test_table_size_tracking(self):
        """Test that table size is tracked correctly."""
        tt = TranspositionTable()
        board = initialize_game_state()
        
        assert len(tt) == 0
        
        # Add some positions
        for i in range(5):
            test_board = board.copy()
            if i < BOARD_COLS:
                apply_player_action(test_board, PlayerAction(i), PLAYER1)
            tt.store_position(test_board, 5, float(i), PlayerAction(i % BOARD_COLS), 'exact')
        
        assert len(tt) > 0
    
    def test_table_size_limit(self):
        """Test that table respects size limits."""
        tt = TranspositionTable(max_table_size=10)
        board = initialize_game_state()
        
        # Add many positions
        for i in range(20):
            test_board = board.copy()
            if i < BOARD_COLS:
                apply_player_action(test_board, PlayerAction(i % BOARD_COLS), PLAYER1)
            if i >= BOARD_COLS:
                apply_player_action(test_board, PlayerAction(i % BOARD_COLS), PLAYER2)
            tt.store_position(test_board, 5, float(i), PlayerAction(i % BOARD_COLS), 'exact')
        
        # Should not exceed max size
        assert len(tt) <= tt.max_table_size
    
    def test_table_replacement_policy(self):
        """Test that table replacement happens correctly."""
        tt = TranspositionTable(max_table_size=5)
        board = initialize_game_state()
        
        # Fill table to capacity with distinct positions
        for i in range(5):
            test_board = board.copy()
            # Create truly different boards that won't be affected by symmetry
            for j in range(i + 1):
                apply_player_action(test_board, PlayerAction(j % BOARD_COLS), PLAYER1)
            tt.store_position(test_board, 5, float(i), PlayerAction(i % BOARD_COLS), 'exact')
        
        initial_size = len(tt)
        assert initial_size <= 5  # Should not exceed max size
        
        # Add one more - should trigger replacement
        test_board = board.copy()
        for j in range(6):  # Make it different from all previous
            apply_player_action(test_board, PlayerAction(j % BOARD_COLS), PLAYER1)
        tt.store_position(test_board, 5, 10.0, PlayerAction(6 % BOARD_COLS), 'exact')
        
        assert len(tt) <= 5


class TestTranspositionTableIntegration:
    """Test transposition table integration with saved state."""
    
    def test_saved_state_transposition_table(self):
        """Test that saved state uses transposition table correctly."""
        saved_state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Store a position through saved state
        saved_state.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
        
        # Look it up
        found, value = saved_state.lookup_position(board, 5, -1000, 1000)
        assert found
        assert value == 10.0
    
    def test_saved_state_board_hash(self):
        """Test that saved state board hashing works."""
        saved_state = MinimaxSavedState()
        board = initialize_game_state()
        
        hash1 = saved_state.get_board_hash(board)
        hash2 = saved_state.get_board_hash(board)
        
        assert hash1 == hash2
    
    def test_saved_state_best_move(self):
        """Test that saved state best move retrieval works."""
        saved_state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Store a position
        saved_state.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
        
        # Get best move
        best_move = saved_state.get_best_move(board)
        assert best_move == PlayerAction(3)
