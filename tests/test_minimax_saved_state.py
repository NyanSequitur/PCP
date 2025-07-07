"""
Unit tests for the minimax saved state functionality.

This module tests the saved state management for the minimax agent,
including transposition table integration and move ordering cache.
"""

import numpy as np
import pytest
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2,
    SavedState
)
from agents.agent_minimax.saved_state import MinimaxSavedState
from agents.agent_minimax.transposition_table import TranspositionTable
from agents.agent_minimax.move_ordering import MoveOrderingCache


class TestMinimaxSavedStateCreation:
    """Test creation and initialization of MinimaxSavedState."""
    
    def test_saved_state_creation_default(self):
        """Test creating saved state with default parameters."""
        state = MinimaxSavedState()
        
        assert isinstance(state, SavedState)
        assert isinstance(state.transposition_table, TranspositionTable)
        assert isinstance(state.move_ordering_cache, MoveOrderingCache)
        assert len(state.transposition_table) == 0
    
    def test_saved_state_creation_custom_size(self):
        """Test creating saved state with custom table size."""
        state = MinimaxSavedState(max_table_size=5000)
        
        assert isinstance(state.transposition_table, TranspositionTable)
        assert state.transposition_table.max_table_size == 5000
    
    def test_saved_state_inheritance(self):
        """Test that MinimaxSavedState properly inherits from SavedState."""
        state = MinimaxSavedState()
        
        assert isinstance(state, SavedState)
        assert isinstance(state, MinimaxSavedState)


class TestMinimaxSavedStateTranspositionTable:
    """Test transposition table integration with saved state."""
    
    def test_store_position_integration(self):
        """Test storing positions through saved state."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Store a position
        state.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
        
        # Verify it was stored
        assert len(state.transposition_table) == 1
        
        # Verify we can look it up
        found, value = state.lookup_position(board, 5, -1000, 1000)
        assert found
        assert value == 10.0
    
    def test_lookup_position_integration(self):
        """Test looking up positions through saved state."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Initially should not find anything
        found, value = state.lookup_position(board, 5, -1000, 1000)
        assert not found
        
        # Store a position
        state.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
        
        # Now should find it
        found, value = state.lookup_position(board, 5, -1000, 1000)
        assert found
        assert value == 10.0
    
    def test_get_best_move_integration(self):
        """Test getting best moves through saved state."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Initially should not find best move
        best_move = state.get_best_move(board)
        assert best_move is None
        
        # Store a position with best move
        state.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
        
        # Should now find best move
        best_move = state.get_best_move(board)
        assert best_move == PlayerAction(3)
    
    def test_get_board_hash_integration(self):
        """Test getting board hash through saved state."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Get hash
        hash1 = state.get_board_hash(board)
        assert isinstance(hash1, str)
        assert len(hash1) > 0
        
        # Hash should be consistent
        hash2 = state.get_board_hash(board)
        assert hash1 == hash2
        
        # Different boards should have different hashes
        apply_player_action(board, PlayerAction(3), PLAYER1)
        hash3 = state.get_board_hash(board)
        assert hash1 != hash3


class TestMinimaxSavedStateMoveOrdering:
    """Test move ordering cache integration with saved state."""
    
    def test_move_ordering_cache_creation(self):
        """Test that move ordering cache is properly created."""
        state = MinimaxSavedState()
        
        assert hasattr(state, 'move_ordering_cache')
        assert isinstance(state.move_ordering_cache, MoveOrderingCache)
    
    def test_move_ordering_cache_functionality(self):
        """Test basic move ordering cache functionality."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Get board hash
        board_hash = state.get_board_hash(board)
        
        # Initial move ordering should be empty
        moves = state.move_ordering_cache.get_move_ordering(board_hash)
        assert len(moves) == 0
        
        # Store some move ordering
        ordered_moves = [PlayerAction(3), PlayerAction(2), PlayerAction(4)]
        state.move_ordering_cache.store_move_ordering(board_hash, ordered_moves)
        
        # Should now return the stored ordering
        moves = state.move_ordering_cache.get_move_ordering(board_hash)
        assert len(moves) == 3
        assert moves == ordered_moves
    
    def test_move_ordering_cache_persistence(self):
        """Test that move ordering cache persists across calls."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Get board hash
        board_hash = state.get_board_hash(board)
        
        # Store move ordering
        ordered_moves = [PlayerAction(3), PlayerAction(2), PlayerAction(4)]
        state.move_ordering_cache.store_move_ordering(board_hash, ordered_moves)
        
        # Should persist across multiple calls
        moves1 = state.move_ordering_cache.get_move_ordering(board_hash)
        moves2 = state.move_ordering_cache.get_move_ordering(board_hash)
        
        assert moves1 == moves2
        assert moves1 == ordered_moves


class TestMinimaxSavedStatePersistence:
    """Test persistence of saved state across moves."""
    
    def test_state_accumulation(self):
        """Test that state accumulates information across moves."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Store multiple positions using different players to avoid symmetry issues
        test_boards = []
        for i in range(5):
            test_board = board.copy()
            # Create distinct boards by placing pieces in different patterns
            if i == 0:
                apply_player_action(test_board, PlayerAction(0), PLAYER1)
            elif i == 1:
                apply_player_action(test_board, PlayerAction(1), PLAYER1)
            elif i == 2:
                apply_player_action(test_board, PlayerAction(2), PLAYER1)
            elif i == 3:
                apply_player_action(test_board, PlayerAction(0), PLAYER1)
                apply_player_action(test_board, PlayerAction(1), PLAYER2)
            elif i == 4:
                apply_player_action(test_board, PlayerAction(3), PLAYER1)
                apply_player_action(test_board, PlayerAction(4), PLAYER2)
            test_boards.append(test_board)
            state.store_position(test_board, 5, float(i), PlayerAction(i), 'exact')
        
        # Should have accumulated entries
        assert len(state.transposition_table) == 5
        
        # All positions should be retrievable
        for i, test_board in enumerate(test_boards):
            found, value = state.lookup_position(test_board, 5, -1000, 1000)
            assert found
            assert value == float(i)
    
    def test_state_reuse_across_games(self):
        """Test that saved state can be reused across different games."""
        state = MinimaxSavedState()
        
        # Store some positions from first game
        board1 = initialize_game_state()
        apply_player_action(board1, PlayerAction(3), PLAYER1)
        state.store_position(board1, 5, 10.0, PlayerAction(4), 'exact')
        
        # Store positions from second game
        board2 = initialize_game_state()
        apply_player_action(board2, PlayerAction(4), PLAYER2)
        state.store_position(board2, 5, 5.0, PlayerAction(3), 'exact')
        
        # Both positions should be available
        found1, value1 = state.lookup_position(board1, 5, -1000, 1000)
        found2, value2 = state.lookup_position(board2, 5, -1000, 1000)
        
        assert found1 and value1 == 10.0
        assert found2 and value2 == 5.0
    
    def test_state_memory_efficiency(self):
        """Test that saved state manages memory efficiently."""
        state = MinimaxSavedState(max_table_size=10)
        board = initialize_game_state()
        
        # Add many positions
        for i in range(20):
            test_board = board.copy()
            # Create different boards by adding pieces
            for j in range(min(i, 6)):
                apply_player_action(test_board, PlayerAction(j), PLAYER1 if j % 2 == 0 else PLAYER2)
            state.store_position(test_board, 5, float(i), PlayerAction(i % 7), 'exact')
        
        # Should not exceed max size
        assert len(state.transposition_table) <= 10


class TestMinimaxSavedStateSymmetry:
    """Test symmetry handling in saved state."""
    
    def test_symmetric_position_handling(self):
        """Test that symmetric positions are handled correctly."""
        state = MinimaxSavedState()
        
        # Create a board and its mirror
        board1 = initialize_game_state()
        apply_player_action(board1, PlayerAction(1), PLAYER1)
        
        board2 = initialize_game_state()
        apply_player_action(board2, PlayerAction(5), PLAYER1)  # Mirror of column 1
        
        # Store position for first board
        state.store_position(board1, 5, 10.0, PlayerAction(2), 'exact')
        
        # Should find mirrored position
        found, value = state.lookup_position(board2, 5, -1000, 1000)
        assert found
        assert value == 10.0
        
        # Best move should be mirrored
        best_move = state.get_best_move(board2)
        assert best_move == PlayerAction(4)  # Mirror of column 2
    
    def test_symmetric_board_hashing(self):
        """Test that symmetric boards have the same hash."""
        state = MinimaxSavedState()
        
        # Create a board and its mirror
        board1 = initialize_game_state()
        apply_player_action(board1, PlayerAction(1), PLAYER1)
        apply_player_action(board1, PlayerAction(2), PLAYER2)
        
        board2 = initialize_game_state()
        apply_player_action(board2, PlayerAction(5), PLAYER1)  # Mirror of column 1
        apply_player_action(board2, PlayerAction(4), PLAYER2)  # Mirror of column 2
        
        # Hashes should be the same
        hash1 = state.get_board_hash(board1)
        hash2 = state.get_board_hash(board2)
        assert hash1 == hash2


class TestMinimaxSavedStateEdgeCases:
    """Test edge cases for saved state functionality."""
    
    def test_empty_board_handling(self):
        """Test handling of empty boards."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Should handle empty board without issues
        hash_val = state.get_board_hash(board)
        assert isinstance(hash_val, str)
        
        best_move = state.get_best_move(board)
        assert best_move is None
        
        found, value = state.lookup_position(board, 5, -1000, 1000)
        assert not found
    
    def test_full_board_handling(self):
        """Test handling of full boards."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Fill the board
        for col in range(7):
            for row in range(6):
                player = PLAYER1 if (col + row) % 2 == 0 else PLAYER2
                apply_player_action(board, PlayerAction(col), player)
        
        # Should handle full board without issues
        hash_val = state.get_board_hash(board)
        assert isinstance(hash_val, str)
        
        # Can store and retrieve positions
        state.store_position(board, 5, 0.0, None, 'exact')
        found, value = state.lookup_position(board, 5, -1000, 1000)
        assert found
        assert value == 0.0
    
    def test_invalid_move_handling(self):
        """Test handling of invalid moves."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Store position with None best move
        state.store_position(board, 5, 10.0, None, 'exact')
        
        # Should handle None best move
        best_move = state.get_best_move(board)
        assert best_move is None
        
        # Should still find the position
        found, value = state.lookup_position(board, 5, -1000, 1000)
        assert found
        assert value == 10.0
    
    def test_state_consistency_after_modifications(self):
        """Test that state remains consistent after modifications."""
        state = MinimaxSavedState()
        board = initialize_game_state()
        
        # Store initial position
        state.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
        
        # Modify the board
        apply_player_action(board, PlayerAction(3), PLAYER1)
        
        # Original position should still be retrievable
        original_board = initialize_game_state()
        found, value = state.lookup_position(original_board, 5, -1000, 1000)
        assert found
        assert value == 10.0
        
        # New position should be different
        new_hash = state.get_board_hash(board)
        old_hash = state.get_board_hash(original_board)
        assert new_hash != old_hash
