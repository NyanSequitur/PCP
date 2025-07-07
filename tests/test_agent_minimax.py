"""
Unit tests for the Minimax agent.
"""
import numpy as np
import pytest
import time
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2, GameState, check_end_state,
    BOARD_COLS
)
from agents.agent_minimax.minimax import generate_move_time_limited, MinimaxSavedState

def test_minimax_winning_move():
    """Test that Minimax finds a winning move for PLAYER1."""
    board = initialize_game_state()
    for c in range(3):
        apply_player_action(board, PlayerAction(c), PLAYER1)
    for c in range(3):
        apply_player_action(board, PlayerAction(c), PLAYER2)
    move, _ = generate_move_time_limited(board, PLAYER1, time_limit_secs=2.0)
    assert move == 3

def test_minimax_blocks_opponent():
    """Test that Minimax blocks opponent's winning move."""
    board = initialize_game_state()
    for c in range(3):
        apply_player_action(board, PlayerAction(c), PLAYER2)
    for c in range(3):
        apply_player_action(board, PlayerAction(c), PLAYER1)
    move, _ = generate_move_time_limited(board, PLAYER1, time_limit_secs=2.0)
    assert move == 3

def test_minimax_draw_raises():
    """Test that Minimax raises when no valid moves (draw)."""
    board = initialize_game_state()
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

def test_minimax_time_limit():
    """Test that Minimax respects the time limit."""
    board = initialize_game_state()
    start = time.time()
    move, _ = generate_move_time_limited(board, PLAYER1, time_limit_secs=1.0)
    elapsed = time.time() - start
    assert elapsed < 2.0

def test_minimax_move_in_range():
    """Test that Minimax returns a move within valid column range."""
    board = initialize_game_state()
    move, _ = generate_move_time_limited(board, PLAYER1, time_limit_secs=1.0)
    assert 0 <= move < 7

def test_minimax_transposition_table_basic():
    """Test that transposition table is created and used."""
    board = initialize_game_state()
    
    # First move - should create a new MinimaxSavedState
    move1, saved_state1 = generate_move_time_limited(board, PLAYER1, None, time_limit_secs=1.0)
    assert isinstance(saved_state1, MinimaxSavedState)
    assert len(saved_state1.transposition_table) > 0
    
    # Apply the move
    apply_player_action(board, move1, PLAYER1)
    
    # Second move - should reuse the transposition table
    move2, saved_state2 = generate_move_time_limited(board, PLAYER2, saved_state1, time_limit_secs=1.0)
    assert isinstance(saved_state2, MinimaxSavedState)
    assert saved_state2 is saved_state1  # Same object
    assert len(saved_state2.transposition_table) >= len(saved_state1.transposition_table)

def test_minimax_transposition_table_persistence():
    """Test that transposition table entries persist and accumulate."""
    board = initialize_game_state()
    
    # Make first move
    move1, state1 = generate_move_time_limited(board, PLAYER1, None, time_limit_secs=1.0)
    assert isinstance(state1, MinimaxSavedState)
    initial_entries = len(state1.transposition_table)
    
    # Apply move and make second move
    apply_player_action(board, move1, PLAYER1)
    move2, state2 = generate_move_time_limited(board, PLAYER2, state1, time_limit_secs=1.0)
    
    # Should have more entries after second search
    assert isinstance(state2, MinimaxSavedState)
    assert len(state2.transposition_table) > initial_entries
    
    # Apply move and make third move
    apply_player_action(board, move2, PLAYER2)
    move3, state3 = generate_move_time_limited(board, PLAYER1, state2, time_limit_secs=1.0)
    
    # Should continue to accumulate entries
    assert isinstance(state3, MinimaxSavedState)
    assert len(state3.transposition_table) >= len(state2.transposition_table)

def test_minimax_transposition_table_best_move():
    """Test that transposition table stores and retrieves best moves."""
    board = initialize_game_state()
    
    # Create a simple position
    apply_player_action(board, PlayerAction(3), PLAYER1)  # Center column
    
    # Generate move and check if best move is stored
    move, state = generate_move_time_limited(board, PLAYER2, None, time_limit_secs=1.0)
    
    # Check that the current position has a best move stored
    assert isinstance(state, MinimaxSavedState)
    best_move = state.get_best_move(board)
    assert best_move is not None
    assert 0 <= int(best_move) < 7

def test_minimax_transposition_table_move_ordering():
    """Test that transposition table improves move ordering."""
    board = initialize_game_state()
    
    # Create a position where center moves are typically better
    apply_player_action(board, PlayerAction(3), PLAYER1)
    apply_player_action(board, PlayerAction(2), PLAYER2)
    apply_player_action(board, PlayerAction(4), PLAYER1)
    
    # First search without saved state
    move1, state1 = generate_move_time_limited(board, PLAYER2, None, time_limit_secs=1.0)
    
    # Apply the move
    apply_player_action(board, move1, PLAYER2)
    
    # Second search should benefit from transposition table
    start_time = time.time()
    move2, state2 = generate_move_time_limited(board, PLAYER1, state1, time_limit_secs=1.0)
    search_time = time.time() - start_time
    
    # The search should complete (no timeout) and be reasonably fast
    assert search_time < 1.5  # Should complete within time limit with some margin

def test_minimax_transposition_table_hash_function():
    """Test that board hashing works correctly."""
    board1 = initialize_game_state()
    board2 = initialize_game_state()
    
    state = MinimaxSavedState()
    
    # Same boards should have same hash
    hash1 = state.get_board_hash(board1)
    hash2 = state.get_board_hash(board2)
    assert hash1 == hash2
    
    # Different boards should have different hashes
    apply_player_action(board2, PlayerAction(3), PLAYER1)
    hash3 = state.get_board_hash(board2)
    assert hash1 != hash3

def test_minimax_transposition_table_position_lookup():
    """Test transposition table position lookup functionality."""
    board = initialize_game_state()
    state = MinimaxSavedState()
    
    # Initially no position should be found
    found, value = state.lookup_position(board, 5, -1000, 1000)
    assert not found
    
    # Store a position
    state.store_position(board, 5, 10.0, PlayerAction(3), 'exact')
    
    # Should now find the position
    found, value = state.lookup_position(board, 5, -1000, 1000)
    assert found
    assert value == 10.0
    
    # Should not find position for deeper search
    found, value = state.lookup_position(board, 6, -1000, 1000)
    assert not found

def test_minimax_symmetry_optimization():
    """Test that board symmetry optimization works correctly."""
    state = MinimaxSavedState()
    
    # Create a board and its mirror
    board = initialize_game_state()
    apply_player_action(board, PlayerAction(1), PLAYER1)  # Left side
    apply_player_action(board, PlayerAction(2), PLAYER2)
    
    mirrored_board = initialize_game_state()
    apply_player_action(mirrored_board, PlayerAction(5), PLAYER1)  # Right side (mirrored)
    apply_player_action(mirrored_board, PlayerAction(4), PLAYER2)
    
    # Both boards should have the same hash (canonical form)
    hash1 = state.get_board_hash(board)
    hash2 = state.get_board_hash(mirrored_board)
    assert hash1 == hash2, "Symmetric boards should have the same hash"
    
    # Store a position for one board
    state.store_position(board, 5, 10.0, PlayerAction(1), 'exact')
    
    # Should be able to lookup the mirrored position
    found, value = state.lookup_position(mirrored_board, 5, -1000, 1000)
    assert found, "Should find mirrored position in transposition table"
    assert value == 10.0, "Should retrieve correct value for mirrored position"
    
    # Best move should be properly mirrored
    best_move_original = state.get_best_move(board)
    best_move_mirrored = state.get_best_move(mirrored_board)
    
    assert best_move_original is not None
    assert best_move_mirrored is not None
    
    # The moves should be mirrors of each other
    assert int(best_move_original) + int(best_move_mirrored) == BOARD_COLS - 1

def test_minimax_symmetry_integration():
    """Test that symmetry optimization improves transposition table efficiency."""
    # Create a symmetric position
    board = initialize_game_state()
    apply_player_action(board, PlayerAction(3), PLAYER1)  # Center
    
    # Make first move
    move1, state1 = generate_move_time_limited(board, PLAYER2, None, time_limit_secs=1.0)
    assert isinstance(state1, MinimaxSavedState)
    initial_entries = len(state1.transposition_table)
    
    # Create the same position but built differently to test symmetry
    board2 = initialize_game_state()
    apply_player_action(board2, PlayerAction(3), PLAYER1)  # Center
    apply_player_action(board2, move1, PLAYER2)
    
    # Mirror the position
    mirrored_board = np.fliplr(board2)
    
    # Search on the mirrored position - should benefit from transposition table
    move2, state2 = generate_move_time_limited(mirrored_board, PLAYER1, state1, time_limit_secs=1.0)
    
    # Should have found entries due to symmetry
    assert isinstance(state2, MinimaxSavedState)
    assert len(state2.transposition_table) >= initial_entries
