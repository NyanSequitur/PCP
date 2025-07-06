"""
Comprehensive test report for the opening book implementation.
This file serves as both a test suite and documentation of the opening book functionality.
"""

import numpy as np
import math
import time
import pytest
from agents.agent_minimax.minimax import MinimaxSavedState, generate_move_time_limited
from game_utils import BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, apply_player_action


def test_opening_book_comprehensive():
    """Comprehensive test of all opening book functionality."""
    
    # Test 1: Board Encoding
    state = MinimaxSavedState()
    empty_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    
    # Test empty board
    empty_key = state._board_to_book_key(empty_board)
    assert len(empty_key) == 42
    assert empty_key == "b" * 42
    
    # Test board with pieces
    test_board = empty_board.copy()
    apply_player_action(test_board, PlayerAction(3), PLAYER1)  # Center column
    apply_player_action(test_board, PlayerAction(2), PLAYER2)  # Left of center
    apply_player_action(test_board, PlayerAction(4), PLAYER1)  # Right of center
    
    key = state._board_to_book_key(test_board)
    
    # Verify column-major, bottom-to-top encoding
    expected_pieces = {
        2: "o",   # PLAYER2 in column 2
        3: "x",   # PLAYER1 in column 3
        4: "x"    # PLAYER1 in column 4
    }
    
    for col, expected_piece in expected_pieces.items():
        actual_piece = key[col * 6]  # First piece in each column
        assert actual_piece == expected_piece
    
    # Test 2: Opening Book Storage and Retrieval
    mock_book = {
        empty_key: 0.0,      # Draw
        key: 15.0            # Advantage position
    }
    
    state.opening_book = mock_book
    state.opening_book_loaded = True
    
    # Test lookups
    result1 = state.lookup_opening_book(empty_board)
    result2 = state.lookup_opening_book(test_board)
    
    assert result1 == 0.0
    assert result2 == 15.0
    
    # Test 3: Position Not in Book
    unknown_board = empty_board.copy()
    apply_player_action(unknown_board, PlayerAction(0), PLAYER1)
    apply_player_action(unknown_board, PlayerAction(6), PLAYER2)
    
    result3 = state.lookup_opening_book(unknown_board)
    assert result3 is None
    
    # Test 4: Integration with Minimax Search
    move, returned_state = generate_move_time_limited(
        empty_board, PLAYER1, state, time_limit_secs=1.0
    )
    
    assert isinstance(move, type(PlayerAction(0)))
    assert isinstance(returned_state, MinimaxSavedState)
    
    # Test 5: Fallback Behavior
    failed_state = MinimaxSavedState()
    failed_state.opening_book_failed = True
    
    move, returned_state = generate_move_time_limited(
        empty_board, PLAYER1, failed_state, time_limit_secs=1.0
    )
    
    assert isinstance(move, type(PlayerAction(0)))
    if isinstance(returned_state, MinimaxSavedState):
        assert returned_state.opening_book_failed
    
    # Test 6: Performance Characteristics
    start_time = time.time()
    for _ in range(1000):  # Reduced for faster testing
        state.lookup_opening_book(empty_board)
    lookup_time = time.time() - start_time
    
    assert lookup_time < 1.0  # Should be very fast
    
    # Test 7: Error Handling
    corrupted_state = MinimaxSavedState()
    corrupted_state.opening_book = {"invalid_key": 999.0}  # Valid float value
    corrupted_state.opening_book_loaded = True
    
    result = corrupted_state.lookup_opening_book(empty_board)
    assert result is None
    
    # Test 8: Different Board States
    test_cases = [
        empty_board,
        test_board,
        unknown_board,
        empty_board.copy()
    ]
    
    # Fill the last board with many pieces
    for i in range(5):
        apply_player_action(test_cases[3], PlayerAction(i), PLAYER1 if i % 2 == 0 else PLAYER2)
    
    for board in test_cases:
        key = state._board_to_book_key(board)
        assert len(key) == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
