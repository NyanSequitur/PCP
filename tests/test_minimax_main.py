"""
Unit tests for the main minimax agent functionality.

This module tests the primary entry points and core functionality of the minimax agent,
including move generation, time limits, and basic behavior.
"""

import numpy as np
import pytest
import time
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2, 
    check_end_state, BOARD_COLS
)
from agents.agent_minimax.minimax import (
    generate_move_time_limited, generate_move, create_minimax_agent
)
from agents.agent_minimax.saved_state import MinimaxSavedState


class TestMinimaxMainFunctionality:
    """Test the main minimax agent functionality."""
    
    def test_generate_move_basic(self):
        """Test basic move generation on empty board."""
        board = initialize_game_state()
        move, saved_state = generate_move(board, PLAYER1)
        
        assert isinstance(move, PlayerAction)
        assert 0 <= move < BOARD_COLS
        assert isinstance(saved_state, MinimaxSavedState)
    
    def test_generate_move_time_limited_basic(self):
        """Test time-limited move generation on empty board."""
        board = initialize_game_state()
        move, saved_state = generate_move_time_limited(
            board, PLAYER1, time_limit_secs=1.0
        )
        
        assert isinstance(move, PlayerAction)
        assert 0 <= move < BOARD_COLS
        assert isinstance(saved_state, MinimaxSavedState)
    
    def test_generate_move_winning_move(self):
        """Test that minimax finds an immediate winning move."""
        board = initialize_game_state()
        
        # Set up a position where PLAYER1 can win by playing column 3
        for c in range(3):
            apply_player_action(board, PlayerAction(c), PLAYER1)
        for c in range(3):
            apply_player_action(board, PlayerAction(c), PLAYER2)
        
        move, _ = generate_move_time_limited(board, PLAYER1, time_limit_secs=2.0)
        assert move == PlayerAction(3)
    
    def test_generate_move_blocks_opponent(self):
        """Test that minimax blocks opponent's winning move."""
        board = initialize_game_state()
        
        # Set up a position where PLAYER2 threatens to win, PLAYER1 must block
        for c in range(3):
            apply_player_action(board, PlayerAction(c), PLAYER2)
        for c in range(3):
            apply_player_action(board, PlayerAction(c), PLAYER1)
        
        move, _ = generate_move_time_limited(board, PLAYER1, time_limit_secs=2.0)
        assert move == PlayerAction(3)
    
    def test_generate_move_no_valid_moves(self):
        """Test behavior when no valid moves are available (full board)."""
        board = initialize_game_state()
        
        # Fill the board completely (draw situation)
        draw_pattern = [
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
            [PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2],
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
            [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
        ]
        for row_idx, row in enumerate(draw_pattern):
            for col_idx, piece in enumerate(row):
                board[5 - row_idx, col_idx] = piece
        
        with pytest.raises(Exception):
            generate_move_time_limited(board, PLAYER1, time_limit_secs=2.0)
    
    def test_generate_move_respects_time_limit(self):
        """Test that move generation respects the time limit."""
        board = initialize_game_state()
        
        start_time = time.time()
        move, _ = generate_move_time_limited(board, PLAYER1, time_limit_secs=1.0)
        elapsed_time = time.time() - start_time
        
        # Should complete well within time limit (allow some buffer)
        assert elapsed_time < 2.0
        assert isinstance(move, PlayerAction)
    
    def test_generate_move_returns_valid_column(self):
        """Test that generated moves are always valid columns."""
        board = initialize_game_state()
        
        for player in [PLAYER1, PLAYER2]:
            move, _ = generate_move_time_limited(board, player, time_limit_secs=1.0)
            assert 0 <= move < BOARD_COLS
    
    def test_generate_move_with_saved_state(self):
        """Test that saved state is properly used and updated."""
        board = initialize_game_state()
        
        # First move without saved state
        move1, state1 = generate_move_time_limited(board, PLAYER1, time_limit_secs=1.0)
        assert isinstance(state1, MinimaxSavedState)
        
        # Apply the move
        apply_player_action(board, move1, PLAYER1)
        
        # Second move with saved state
        move2, state2 = generate_move_time_limited(board, PLAYER2, state1, time_limit_secs=1.0)
        assert isinstance(state2, MinimaxSavedState)
        assert state2 is state1  # Should be the same object


class TestMinimaxAgentCreation:
    """Test the minimax agent creation and configuration."""
    
    def test_create_minimax_agent_default(self):
        """Test creating minimax agent with default parameters."""
        move_function, initial_state = create_minimax_agent()
        
        assert callable(move_function)
        assert isinstance(initial_state, MinimaxSavedState)
    
    def test_create_minimax_agent_custom_params(self):
        """Test creating minimax agent with custom parameters."""
        move_function, initial_state = create_minimax_agent(
            time_limit=3.0,
            max_depth=15,
            max_table_size=500000
        )
        
        assert callable(move_function)
        assert isinstance(initial_state, MinimaxSavedState)
        assert initial_state.transposition_table.max_table_size == 500000
    
    def test_created_agent_functionality(self):
        """Test that created agent actually works."""
        move_function, initial_state = create_minimax_agent(time_limit=1.0)
        
        board = initialize_game_state()
        move, updated_state = move_function(board, PLAYER1, initial_state)
        
        assert isinstance(move, PlayerAction)
        assert 0 <= move < BOARD_COLS
        assert isinstance(updated_state, MinimaxSavedState)


class TestMinimaxInputValidation:
    """Test input validation for minimax functions."""
    
    def test_invalid_time_limit(self):
        """Test that invalid time limits raise appropriate errors."""
        board = initialize_game_state()
        
        with pytest.raises(ValueError, match="Time limit must be positive"):
            generate_move_time_limited(board, PLAYER1, time_limit_secs=0.0)
        
        with pytest.raises(ValueError, match="Time limit must be positive"):
            generate_move_time_limited(board, PLAYER1, time_limit_secs=-1.0)
    
    def test_invalid_max_depth(self):
        """Test that invalid max depth raises appropriate errors."""
        board = initialize_game_state()
        
        with pytest.raises(ValueError, match="Maximum depth must be positive"):
            generate_move_time_limited(board, PLAYER1, max_depth=0)
        
        with pytest.raises(ValueError, match="Maximum depth must be positive"):
            generate_move_time_limited(board, PLAYER1, max_depth=-1)
    
    def test_invalid_player(self):
        """Test that invalid player values raise appropriate errors."""
        board = initialize_game_state()
        
        with pytest.raises(ValueError, match="Player must be PLAYER1 or PLAYER2"):
            generate_move_time_limited(board, 0, time_limit_secs=1.0)
        
        with pytest.raises(ValueError, match="Player must be PLAYER1 or PLAYER2"):
            generate_move_time_limited(board, 3, time_limit_secs=1.0)
