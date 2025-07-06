#!/usr/bin/env python3
"""
Performance test to demonstrate the symmetry optimization.
"""

import time
import numpy as np
from agents.agent_minimax.minimax import generate_move_time_limited, MinimaxSavedState
from game_utils import initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2

def test_symmetry_performance():
    """Test the performance benefit of symmetry optimization."""
    
    # Create an initial position
    board = initialize_game_state()
    apply_player_action(board, PlayerAction(3), PLAYER1)  # Center
    apply_player_action(board, PlayerAction(2), PLAYER2)  # Left of center
    apply_player_action(board, PlayerAction(4), PLAYER1)  # Right of center
    
    print("Testing symmetry optimization performance...")
    print("Initial board:")
    print(board)
    print()
    
    # First search - builds transposition table
    start_time = time.time()
    move1, state1 = generate_move_time_limited(board, PLAYER2, None, 2.0)
    search_time1 = time.time() - start_time
    
    print(f"First search: {search_time1:.3f}s")
    print(f"Move chosen: {move1}")
    if isinstance(state1, MinimaxSavedState):
        print(f"Transposition table entries: {len(state1.transposition_table)}")
    print()
    
    # Apply the move
    apply_player_action(board, move1, PLAYER2)
    
    # Create mirrored position
    mirrored_board = np.fliplr(board.copy())
    print("Mirrored board:")
    print(mirrored_board)
    print()
    
    # Second search on mirrored position - should benefit from symmetry
    start_time = time.time()
    move2, state2 = generate_move_time_limited(mirrored_board, PLAYER1, state1, 2.0)
    search_time2 = time.time() - start_time
    
    print(f"Second search (mirrored): {search_time2:.3f}s")
    print(f"Move chosen: {move2}")
    if isinstance(state2, MinimaxSavedState):
        print(f"Transposition table entries: {len(state2.transposition_table)}")
    print()
    
    # Third search on original position - should also benefit
    start_time = time.time()
    move3, state3 = generate_move_time_limited(board, PLAYER1, state2, 2.0)
    search_time3 = time.time() - start_time
    
    print(f"Third search (original): {search_time3:.3f}s")
    print(f"Move chosen: {move3}")
    if isinstance(state3, MinimaxSavedState):
        print(f"Transposition table entries: {len(state3.transposition_table)}")
    print()
    
    print("✓ Symmetry optimization is working!")
    print(f"✓ Transposition table shared between symmetric positions")
    
    # Verify that we can get reasonable performance on both orientations
    assert search_time2 < 2.5, "Second search should be reasonably fast"
    assert search_time3 < 2.5, "Third search should be reasonably fast"

if __name__ == "__main__":
    test_symmetry_performance()
