"""
Test script to verify that the symmetry improvement in minimax agent is working correctly.

This script tests:
1. Canonical board generation
2. Hash consistency for symmetric positions
3. Transposition table sharing between symmetric positions
4. Performance improvements from symmetry
"""

import numpy as np
import time
from typing import Tuple
from agents.agent_minimax.minimax import (
    MinimaxSavedState, generate_move_time_limited, 
    _get_valid_columns, _negamax, _other
)
from game_utils import (
    BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2,
    apply_player_action, PlayerAction, BoardPiece
)


def create_test_board() -> np.ndarray:
    """Create a test board with some pieces."""
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=BoardPiece)
    
    # Add some pieces to create an asymmetric position
    board[5, 1] = PLAYER1  # Column 1, bottom row
    board[5, 2] = PLAYER2  # Column 2, bottom row
    board[4, 1] = PLAYER2  # Column 1, second row
    board[5, 4] = PLAYER1  # Column 4, bottom row
    
    return board


def create_symmetric_board() -> np.ndarray:
    """Create a symmetric test board."""
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=BoardPiece)
    
    # Create a symmetric position
    center = BOARD_COLS // 2
    board[5, center] = PLAYER1     # Center column
    board[5, center-1] = PLAYER2   # Left of center
    board[5, center+1] = PLAYER2   # Right of center
    board[4, center] = PLAYER2     # Center column, second row
    
    return board


def mirror_board(board: np.ndarray) -> np.ndarray:
    """Mirror a board horizontally."""
    return np.fliplr(board)


def test_canonical_board_generation():
    """Test that canonical board generation works correctly."""
    print("Testing canonical board generation...")
    
    state = MinimaxSavedState()
    
    # Test with asymmetric board
    board = create_test_board()
    mirrored = mirror_board(board)
    
    canonical1 = state._get_canonical_board(board)
    canonical2 = state._get_canonical_board(mirrored)
    
    print(f"Original board:\n{board}")
    print(f"Mirrored board:\n{mirrored}")
    print(f"Canonical 1:\n{canonical1}")
    print(f"Canonical 2:\n{canonical2}")
    
    # Both should produce the same canonical form
    assert np.array_equal(canonical1, canonical2), "Canonical forms should be identical"
    print("✓ Canonical board generation works correctly")


def test_hash_consistency():
    """Test that symmetric positions produce the same hash."""
    print("\nTesting hash consistency for symmetric positions...")
    
    state = MinimaxSavedState()
    
    # Test with asymmetric board
    board = create_test_board()
    mirrored = mirror_board(board)
    
    hash1 = state.get_board_hash(board)
    hash2 = state.get_board_hash(mirrored)
    
    print(f"Hash of original board: {hash1}")
    print(f"Hash of mirrored board: {hash2}")
    
    assert hash1 == hash2, "Hashes should be identical for symmetric positions"
    print("✓ Hash consistency works correctly")


def test_transposition_table_sharing():
    """Test that transposition table entries are shared between symmetric positions."""
    print("\nTesting transposition table sharing...")
    
    state = MinimaxSavedState()
    
    # Create asymmetric board
    board = create_test_board()
    mirrored = mirror_board(board)
    
    # Store a position for the original board
    best_move = PlayerAction(2)
    state.store_position(board, depth=3, value=5.0, best_move=best_move, flag='exact')
    
    # Check if we can retrieve it for the mirrored board
    found, value = state.lookup_position(mirrored, depth=3, alpha=-100, beta=100)
    
    print(f"Stored position for original board with best move: {best_move}")
    print(f"Lookup on mirrored board - Found: {found}, Value: {value}")
    
    assert found, "Should find the stored position for mirrored board"
    assert value == 5.0, "Should retrieve the same value"
    print("✓ Transposition table sharing works correctly")


def test_move_mirroring():
    """Test that best moves are correctly mirrored."""
    print("\nTesting move mirroring...")
    
    state = MinimaxSavedState()
    
    # Create asymmetric board
    board = create_test_board()
    mirrored = mirror_board(board)
    
    # Store a position for the original board
    best_move = PlayerAction(1)  # Column 1
    state.store_position(board, depth=3, value=5.0, best_move=best_move, flag='exact')
    
    # Get best move for original board
    retrieved_move1 = state.get_best_move(board)
    print(f"Best move for original board: {retrieved_move1}")
    
    # Get best move for mirrored board (should be mirrored)
    retrieved_move2 = state.get_best_move(mirrored)
    print(f"Best move for mirrored board: {retrieved_move2}")
    
    # Column 1 should mirror to column 5 (BOARD_COLS - 1 - 1 = 5)
    expected_mirrored = PlayerAction(BOARD_COLS - 1 - 1)
    
    assert retrieved_move1 is not None, "Should retrieve a move for original board"
    assert retrieved_move2 is not None, "Should retrieve a move for mirrored board"
    
    print(f"Original board is mirrored: {state._is_board_mirrored(board)}")
    print(f"Mirrored board is mirrored: {state._is_board_mirrored(mirrored)}")
    
    # Debug the logic
    print(f"Expected mirrored move: {expected_mirrored}")
    print(f"Retrieved move 1: {retrieved_move1}")
    print(f"Retrieved move 2: {retrieved_move2}")
    
    # The logic should be:
    # - Both boards should have the same canonical form
    # - The moves should be mirrored relative to each other
    # - If we store move 1 for original board, we should get move 1 for original and move 5 for mirrored
    expected_original = best_move
    expected_for_mirrored = PlayerAction(BOARD_COLS - 1 - int(best_move))
    
    assert int(retrieved_move1) == int(expected_original), f"Expected {expected_original}, got {retrieved_move1}"
    assert int(retrieved_move2) == int(expected_for_mirrored), f"Expected {expected_for_mirrored}, got {retrieved_move2}"
    
    print("✓ Move mirroring works correctly")


def test_performance_improvement():
    """Test that symmetry provides performance improvement."""
    print("\nTesting performance improvement from symmetry...")
    
    # Create a position that will have many symmetric variations
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=BoardPiece)
    
    # Add a few pieces to create interesting positions
    board[5, 2] = PLAYER1
    board[5, 3] = PLAYER2
    board[4, 3] = PLAYER1
    
    # Test with symmetry enabled
    print("Testing with symmetry enabled...")
    start_time = time.time()
    
    # Generate moves for both the original and mirrored positions
    # This should benefit from transposition table sharing
    move1, state1 = generate_move_time_limited(board, PLAYER1, None, time_limit_secs=1.0)
    mirrored_board = mirror_board(board)
    move2, state2 = generate_move_time_limited(mirrored_board, PLAYER1, state1, time_limit_secs=1.0)
    
    time_with_symmetry = time.time() - start_time
    
    print(f"Original board move: {move1}")
    print(f"Mirrored board move: {move2}")
    print(f"Time with symmetry: {time_with_symmetry:.3f} seconds")
    
    # Cast to MinimaxSavedState for type safety
    if isinstance(state2, MinimaxSavedState):
        print(f"Transposition table size: {len(state2.transposition_table)}")
    else:
        print("Warning: state2 is not MinimaxSavedState")
    
    # Test without symmetry (create new state for each)
    print("\nTesting without symmetry (separate states)...")
    start_time = time.time()
    
    move3, state3 = generate_move_time_limited(board, PLAYER1, None, time_limit_secs=1.0)
    move4, state4 = generate_move_time_limited(mirrored_board, PLAYER1, None, time_limit_secs=1.0)
    
    time_without_symmetry = time.time() - start_time
    
    print(f"Time without symmetry: {time_without_symmetry:.3f} seconds")
    
    # Calculate table sizes safely
    size3 = len(state3.transposition_table) if isinstance(state3, MinimaxSavedState) else 0
    size4 = len(state4.transposition_table) if isinstance(state4, MinimaxSavedState) else 0
    print(f"Total transposition table entries: {size3 + size4}")
    
    # With symmetry, we should have fewer total table entries and potentially better performance
    total_entries_with_symmetry = len(state2.transposition_table) if isinstance(state2, MinimaxSavedState) else 0
    total_entries_without_symmetry = size3 + size4
    
    print(f"\nTransposition table entries with symmetry: {total_entries_with_symmetry}")
    print(f"Transposition table entries without symmetry: {total_entries_without_symmetry}")
    
    if total_entries_with_symmetry < total_entries_without_symmetry:
        print("✓ Symmetry reduces transposition table size")
    else:
        print("? Symmetry didn't reduce transposition table size in this test")
    
    # The moves should be related by symmetry
    expected_mirrored_move = PlayerAction(BOARD_COLS - 1 - int(move1))
    if int(move2) == int(expected_mirrored_move):
        print("✓ Symmetric positions produce symmetric moves")
    else:
        print(f"? Moves not perfectly symmetric: {move1} -> expected {expected_mirrored_move}, got {move2}")


def test_edge_cases():
    """Test edge cases for symmetry handling."""
    print("\nTesting edge cases...")
    
    state = MinimaxSavedState()
    
    # Test empty board (should be symmetric)
    empty_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=BoardPiece)
    canonical = state._get_canonical_board(empty_board)
    
    assert np.array_equal(canonical, empty_board), "Empty board should be its own canonical form"
    print("✓ Empty board handled correctly")
    
    # Test perfectly symmetric board
    symmetric_board = create_symmetric_board()
    canonical = state._get_canonical_board(symmetric_board)
    
    assert np.array_equal(canonical, symmetric_board), "Symmetric board should be its own canonical form"
    print("✓ Symmetric board handled correctly")
    
    # Test center column moves
    center_col = BOARD_COLS // 2
    center_move = PlayerAction(center_col)
    mirrored_center = state._mirror_move(center_move)
    
    assert int(mirrored_center) == center_col, "Center column should mirror to itself"
    print("✓ Center column mirroring handled correctly")


def main():
    """Run all symmetry tests."""
    print("Testing minimax agent symmetry improvements...")
    print("=" * 60)
    
    test_canonical_board_generation()
    test_hash_consistency()
    test_transposition_table_sharing()
    test_move_mirroring()
    test_performance_improvement()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("All symmetry tests completed!")


if __name__ == "__main__":
    main()
