"""
Integration test for opening book functionality with the minimax agent.
This test demonstrates the opening book working with the actual minimax search.
"""

import numpy as np
import math
import pytest
from unittest.mock import Mock, patch, MagicMock
from agents.agent_minimax.minimax import generate_move_time_limited, MinimaxSavedState
from game_utils import BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, apply_player_action


def test_opening_book_integration():
    """Test opening book integration with the minimax agent."""
    
    # Create a mock opening book with some test positions
    mock_opening_book = {}
    
    # Empty board position (should be a draw)
    empty_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    state = MinimaxSavedState()
    empty_key = state._board_to_book_key(empty_board)
    mock_opening_book[empty_key] = 0.0  # Draw
    
    # Position with move in center column (should be winning for first player)
    center_board = empty_board.copy()
    apply_player_action(center_board, PlayerAction(3), PLAYER1)
    center_key = state._board_to_book_key(center_board)
    mock_opening_book[center_key] = math.inf  # Win for current player
    
    # Position with move in corner column (should be losing for first player)
    corner_board = empty_board.copy()
    apply_player_action(corner_board, PlayerAction(0), PLAYER1)
    corner_key = state._board_to_book_key(corner_board)
    mock_opening_book[corner_key] = -math.inf  # Loss for current player
    
    # Create a saved state with the mock opening book
    saved_state = MinimaxSavedState()
    saved_state.opening_book = mock_opening_book
    saved_state.opening_book_loaded = True
    
    # Test 1: Empty board should use opening book (draw value)
    move, returned_state = generate_move_time_limited(
        empty_board, PLAYER1, saved_state, time_limit_secs=1.0
    )
    assert isinstance(move, type(PlayerAction(0)))
    assert isinstance(returned_state, MinimaxSavedState)
    
    # Test 2: Verify opening book positions are found
    result1 = saved_state.lookup_opening_book(empty_board)
    result2 = saved_state.lookup_opening_book(center_board)
    result3 = saved_state.lookup_opening_book(corner_board)
    
    assert result1 == 0.0
    assert result2 == math.inf
    assert result3 == -math.inf
    
    # Test 3: Position not in opening book should return None
    non_book_board = empty_board.copy()
    apply_player_action(non_book_board, PlayerAction(1), PLAYER1)
    apply_player_action(non_book_board, PlayerAction(2), PLAYER2)
    
    result4 = saved_state.lookup_opening_book(non_book_board)
    assert result4 is None
    
    # Test 4: Still generates valid moves for non-book positions
    move, returned_state = generate_move_time_limited(
        non_book_board, PLAYER1, saved_state, time_limit_secs=1.0
    )
    assert isinstance(move, type(PlayerAction(0)))
    assert isinstance(returned_state, MinimaxSavedState)


def test_opening_book_with_real_uci_format():
    """Test opening book with positions in real UCI format."""
    
    # Create a position that would be in the UCI dataset format
    # This represents a simple opening sequence
    state = MinimaxSavedState()
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    
    # Player 1 plays in center (column 3)
    apply_player_action(board, PlayerAction(3), PLAYER1)
    
    # Player 2 plays in column 2
    apply_player_action(board, PlayerAction(2), PLAYER2)
    
    # Player 1 plays in column 4
    apply_player_action(board, PlayerAction(4), PLAYER1)
    
    # Get the UCI format key
    uci_key = state._board_to_book_key(board)
    
    # Verify the format is correct (should be 42 characters, column-major, bottom-to-top)
    assert len(uci_key) == 42
    
    # Check that the pieces are in the right positions
    # Column 0: all empty -> "bbbbbb"
    # Column 1: all empty -> "bbbbbb"  
    # Column 2: PLAYER2 at bottom -> "obbbbb"
    # Column 3: PLAYER1 at bottom -> "xbbbbb"
    # Column 4: PLAYER1 at bottom -> "xbbbbb"
    # Columns 5-6: all empty -> "bbbbbb" each
    
    expected_pattern = "bbbbbb" + "bbbbbb" + "obbbbb" + "xbbbbb" + "xbbbbb" + "bbbbbb" + "bbbbbb"
    assert uci_key == expected_pattern
    
    # Test with a mock opening book containing this position
    mock_book = {uci_key: 5.0}  # Advantage to current player
    state.opening_book = mock_book
    state.opening_book_loaded = True
    
    # Lookup should find the position
    result = state.lookup_opening_book(board)
    assert result == 5.0


def test_opening_book_error_handling():
    """Test opening book error handling and fallback behavior."""
    
    # Test with a saved state that has opening book loading disabled
    saved_state = MinimaxSavedState()
    saved_state.opening_book_failed = True
    
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    
    # Should still work without opening book
    move, returned_state = generate_move_time_limited(
        board, PLAYER1, saved_state, time_limit_secs=1.0
    )
    
    assert isinstance(move, type(PlayerAction(0)))
    assert isinstance(returned_state, MinimaxSavedState)
    assert returned_state.opening_book_failed


def test_opening_book_symmetry():
    """Test opening book handling of symmetric positions."""
    
    state = MinimaxSavedState()
    
    # Create a symmetric position
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    apply_player_action(board, PlayerAction(1), PLAYER1)  # Left side
    
    # Create the mirror position
    mirror_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    apply_player_action(mirror_board, PlayerAction(5), PLAYER1)  # Right side (mirror of column 1)
    
    # Both positions should generate the same key due to canonical board handling
    key1 = state.get_board_hash(board)
    key2 = state.get_board_hash(mirror_board)
    
    # The keys should be the same due to canonical representation
    assert key1 == key2
    
    # Test that opening book keys are consistent
    book_key1 = state._board_to_book_key(state._get_canonical_board(board))
    book_key2 = state._board_to_book_key(state._get_canonical_board(mirror_board))
    assert book_key1 == book_key2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
