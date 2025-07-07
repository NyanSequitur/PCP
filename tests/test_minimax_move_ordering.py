"""
Unit tests for the minimax move ordering functionality.

This module tests the move ordering utilities used to improve alpha-beta pruning
efficiency in the minimax search.
"""

import numpy as np
import pytest
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2,
    BOARD_COLS
)
from agents.agent_minimax.move_ordering import (
    get_valid_columns, order_moves, MoveOrderingCache
)
from agents.agent_minimax.transposition_table import TranspositionTable


class TestGetValidColumns:
    """Test getting valid columns for moves."""
    
    def test_get_valid_columns_empty_board(self):
        """Test getting valid columns from empty board."""
        board = initialize_game_state()
        valid_cols = get_valid_columns(board)
        
        # All columns should be valid on empty board
        assert len(valid_cols) == BOARD_COLS
        assert valid_cols == list(range(BOARD_COLS))
    
    def test_get_valid_columns_partially_filled(self):
        """Test getting valid columns from partially filled board."""
        board = initialize_game_state()
        
        # Fill first column completely
        for _ in range(6):
            apply_player_action(board, PlayerAction(0), PLAYER1)
        
        valid_cols = get_valid_columns(board)
        
        # First column should not be valid
        assert 0 not in valid_cols
        assert len(valid_cols) == BOARD_COLS - 1
        assert valid_cols == list(range(1, BOARD_COLS))
    
    def test_get_valid_columns_multiple_filled(self):
        """Test getting valid columns when multiple columns are filled."""
        board = initialize_game_state()
        
        # Fill columns 0, 2, 4
        for col in [0, 2, 4]:
            for _ in range(6):
                apply_player_action(board, PlayerAction(col), PLAYER1)
        
        valid_cols = get_valid_columns(board)
        
        # Should have columns 1, 3, 5, 6
        expected = [1, 3, 5, 6]
        assert valid_cols == expected
    
    def test_get_valid_columns_almost_full_board(self):
        """Test getting valid columns when board is almost full."""
        board = initialize_game_state()
        
        # Fill all columns except the last one
        for col in range(BOARD_COLS - 1):
            for _ in range(6):
                apply_player_action(board, PlayerAction(col), PLAYER1)
        
        valid_cols = get_valid_columns(board)
        
        # Only last column should be valid
        assert valid_cols == [BOARD_COLS - 1]
    
    def test_get_valid_columns_full_board(self):
        """Test getting valid columns when board is completely full."""
        board = initialize_game_state()
        
        # Fill all columns
        for col in range(BOARD_COLS):
            for _ in range(6):
                player = PLAYER1 if col % 2 == 0 else PLAYER2
                apply_player_action(board, PlayerAction(col), player)
        
        valid_cols = get_valid_columns(board)
        
        # No columns should be valid
        assert valid_cols == []


class TestOrderMoves:
    """Test move ordering functionality."""
    
    def test_order_moves_empty_board(self):
        """Test move ordering on empty board."""
        board = initialize_game_state()
        valid_cols = get_valid_columns(board)
        tt = TranspositionTable()
        
        ordered_cols = order_moves(board, valid_cols, tt)
        
        # Should return all valid columns
        assert len(ordered_cols) == len(valid_cols)
        assert all(col in valid_cols for col in ordered_cols)
        
        # Should prioritize center columns
        assert ordered_cols[0] == 3  # Center column should be first
    
    def test_order_moves_with_transposition_table(self):
        """Test move ordering with transposition table best move."""
        board = initialize_game_state()
        valid_cols = get_valid_columns(board)
        tt = TranspositionTable()
        
        # Store a best move in transposition table
        tt.store_position(board, 5, 10.0, PlayerAction(5), 'exact')
        
        ordered_cols = order_moves(board, valid_cols, tt)
        
        # Should prioritize the best move from transposition table
        assert ordered_cols[0] == 5
        assert len(ordered_cols) == len(valid_cols)
    
    def test_order_moves_center_preference(self):
        """Test that move ordering prefers center columns."""
        board = initialize_game_state()
        valid_cols = get_valid_columns(board)
        tt = TranspositionTable()
        
        ordered_cols = order_moves(board, valid_cols, tt)
        
        # Center column (3) should be first
        assert ordered_cols[0] == 3
        
        # Should generally prefer columns closer to center
        center_distance = lambda col: abs(col - 3)
        for i in range(len(ordered_cols) - 1):
            # Allow some flexibility in ordering but check general trend
            assert center_distance(ordered_cols[i]) <= center_distance(ordered_cols[i + 1]) + 1
    
    def test_order_moves_partial_valid_columns(self):
        """Test move ordering with only some columns valid."""
        board = initialize_game_state()
        tt = TranspositionTable()
        
        # Fill some columns
        for col in [0, 1, 6]:
            for _ in range(6):
                apply_player_action(board, PlayerAction(col), PLAYER1)
        
        valid_cols = get_valid_columns(board)
        ordered_cols = order_moves(board, valid_cols, tt)
        
        # Should only include valid columns
        assert len(ordered_cols) == len(valid_cols)
        assert all(col in valid_cols for col in ordered_cols)
        
        # Should still prioritize center if available
        if 3 in valid_cols:
            assert ordered_cols[0] == 3
    
    def test_order_moves_consistency(self):
        """Test that move ordering is consistent for same position."""
        board = initialize_game_state()
        valid_cols = get_valid_columns(board)
        tt = TranspositionTable()
        
        # Apply some moves
        apply_player_action(board, PlayerAction(3), PLAYER1)
        apply_player_action(board, PlayerAction(2), PLAYER2)
        
        valid_cols = get_valid_columns(board)
        
        # Order moves multiple times
        ordered_cols1 = order_moves(board, valid_cols, tt)
        ordered_cols2 = order_moves(board, valid_cols, tt)
        
        # Should be consistent
        assert ordered_cols1 == ordered_cols2
    
    def test_order_moves_with_invalid_tt_move(self):
        """Test move ordering when transposition table has invalid move."""
        board = initialize_game_state()
        tt = TranspositionTable()
        
        # Fill column 5 completely
        for _ in range(6):
            apply_player_action(board, PlayerAction(5), PLAYER1)
        
        # Store invalid move (column 5) in transposition table
        tt.store_position(board, 5, 10.0, PlayerAction(5), 'exact')
        
        valid_cols = get_valid_columns(board)
        ordered_cols = order_moves(board, valid_cols, tt)
        
        # Should not include the invalid move
        assert 5 not in ordered_cols
        assert len(ordered_cols) == len(valid_cols)
        assert all(col in valid_cols for col in ordered_cols)


class TestMoveOrderingCache:
    """Test the move ordering cache."""
    
    def test_move_ordering_cache_creation(self):
        """Test creating a move ordering cache."""
        cache = MoveOrderingCache()
        
        assert isinstance(cache.cache, dict)
        assert len(cache.cache) == 0
    
    def test_move_ordering_cache_store_and_get(self):
        """Test storing and retrieving move orderings."""
        cache = MoveOrderingCache()
        board_hash = "test_hash"
        moves = [PlayerAction(3), PlayerAction(2), PlayerAction(4)]
        
        # Initially should return empty list
        retrieved = cache.get_move_ordering(board_hash)
        assert retrieved == []
        
        # Store moves
        cache.store_move_ordering(board_hash, moves)
        
        # Should now return stored moves
        retrieved = cache.get_move_ordering(board_hash)
        assert retrieved == moves
    
    def test_move_ordering_cache_multiple_positions(self):
        """Test cache with multiple positions."""
        cache = MoveOrderingCache()
        
        # Store multiple positions
        positions = {
            "hash1": [PlayerAction(3), PlayerAction(2)],
            "hash2": [PlayerAction(4), PlayerAction(5)],
            "hash3": [PlayerAction(1), PlayerAction(6)]
        }
        
        for hash_key, moves in positions.items():
            cache.store_move_ordering(hash_key, moves)
        
        # Should retrieve all correctly
        for hash_key, expected_moves in positions.items():
            retrieved = cache.get_move_ordering(hash_key)
            assert retrieved == expected_moves
    
    def test_move_ordering_cache_update_existing(self):
        """Test updating existing move ordering."""
        cache = MoveOrderingCache()
        board_hash = "test_hash"
        
        # Store initial moves
        initial_moves = [PlayerAction(3), PlayerAction(2)]
        cache.store_move_ordering(board_hash, initial_moves)
        
        # Update with new moves
        new_moves = [PlayerAction(4), PlayerAction(5), PlayerAction(6)]
        cache.store_move_ordering(board_hash, new_moves)
        
        # Should return updated moves
        retrieved = cache.get_move_ordering(board_hash)
        assert retrieved == new_moves
    
    def test_move_ordering_cache_empty_moves(self):
        """Test cache with empty move list."""
        cache = MoveOrderingCache()
        board_hash = "test_hash"
        
        # Store empty moves
        cache.store_move_ordering(board_hash, [])
        
        # Should return empty list
        retrieved = cache.get_move_ordering(board_hash)
        assert retrieved == []
    
    def test_move_ordering_cache_nonexistent_hash(self):
        """Test cache with nonexistent hash."""
        cache = MoveOrderingCache()
        
        # Should return empty list for nonexistent hash
        retrieved = cache.get_move_ordering("nonexistent")
        assert retrieved == []


class TestMoveOrderingIntegration:
    """Test integration between move ordering components."""
    
    def test_move_ordering_with_game_progression(self):
        """Test move ordering as game progresses."""
        board = initialize_game_state()
        tt = TranspositionTable()
        
        # Test ordering at different stages of the game
        stages = []
        
        # Stage 1: Empty board
        valid_cols = get_valid_columns(board)
        ordered_cols = order_moves(board, valid_cols, tt)
        stages.append(ordered_cols)
        
        # Stage 2: After one move
        apply_player_action(board, PlayerAction(3), PLAYER1)
        valid_cols = get_valid_columns(board)
        ordered_cols = order_moves(board, valid_cols, tt)
        stages.append(ordered_cols)
        
        # Stage 3: After multiple moves
        apply_player_action(board, PlayerAction(2), PLAYER2)
        apply_player_action(board, PlayerAction(4), PLAYER1)
        valid_cols = get_valid_columns(board)
        ordered_cols = order_moves(board, valid_cols, tt)
        stages.append(ordered_cols)
        
        # All stages should produce valid orderings
        for stage in stages:
            assert len(stage) > 0
            assert all(isinstance(col, int) for col in stage)
    
    def test_move_ordering_with_symmetry(self):
        """Test move ordering with symmetric positions."""
        board1 = initialize_game_state()
        board2 = initialize_game_state()
        tt = TranspositionTable()
        
        # Create symmetric positions
        apply_player_action(board1, PlayerAction(1), PLAYER1)
        apply_player_action(board2, PlayerAction(5), PLAYER1)  # Mirror of column 1
        
        valid_cols1 = get_valid_columns(board1)
        valid_cols2 = get_valid_columns(board2)
        
        ordered_cols1 = order_moves(board1, valid_cols1, tt)
        ordered_cols2 = order_moves(board2, valid_cols2, tt)
        
        # Both should produce valid orderings
        assert len(ordered_cols1) > 0
        assert len(ordered_cols2) > 0
        assert all(col in valid_cols1 for col in ordered_cols1)
        assert all(col in valid_cols2 for col in ordered_cols2)
    
    def test_move_ordering_with_filled_columns(self):
        """Test move ordering as columns get filled."""
        board = initialize_game_state()
        tt = TranspositionTable()
        
        # Track how ordering changes as columns are filled
        orderings = []
        
        for fill_col in range(BOARD_COLS):
            # Fill the column
            for _ in range(6):
                apply_player_action(board, PlayerAction(fill_col), PLAYER1)
            
            # Get ordering
            valid_cols = get_valid_columns(board)
            if valid_cols:  # Only test if there are valid columns
                ordered_cols = order_moves(board, valid_cols, tt)
                orderings.append(ordered_cols)
                
                # Should only include valid columns
                assert all(col in valid_cols for col in ordered_cols)
                assert len(ordered_cols) == len(valid_cols)
        
        # Should have progressively fewer columns
        for i in range(len(orderings) - 1):
            assert len(orderings[i]) >= len(orderings[i + 1])
    
    def test_move_ordering_performance_consistency(self):
        """Test that move ordering is efficient and consistent."""
        board = initialize_game_state()
        tt = TranspositionTable()
        
        # Apply some moves
        moves = [3, 2, 4, 1, 5, 0, 6]
        for i, col in enumerate(moves):
            if i < 4:  # Don't fill the board completely
                apply_player_action(board, PlayerAction(col), PLAYER1 if i % 2 == 0 else PLAYER2)
        
        valid_cols = get_valid_columns(board)
        
        # Test multiple orderings
        orderings = []
        for _ in range(10):
            ordered_cols = order_moves(board, valid_cols, tt)
            orderings.append(ordered_cols)
        
        # All orderings should be identical
        for ordering in orderings:
            assert ordering == orderings[0]
