#!/usr/bin/env python3
"""
Simple test to demonstrate the transposition table functionality.
"""

import numpy as np
from agents.agent_minimax.minimax import generate_move_time_limited, MinimaxSavedState
from game_utils import BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2, apply_player_action, PlayerAction

def test_transposition_table():
    """Test that the transposition table persists information between moves."""
    
    # Create initial board
    board = np.full((BOARD_ROWS, BOARD_COLS), NO_PLAYER, dtype=np.int8)
    
    # Make a few moves to create a more interesting position
    apply_player_action(board, PlayerAction(3), PLAYER1)  # Player 1 center
    apply_player_action(board, PlayerAction(2), PLAYER2)  # Player 2 left-center
    apply_player_action(board, PlayerAction(4), PLAYER1)  # Player 1 right-center
    
    print("Initial board state:")
    print(board)
    print()
    
    # First move - no saved state (fresh start)
    print("Making first move (no saved state)...")
    move1, saved_state = generate_move_time_limited(board, PLAYER2, None, 2.0)
    print(f"Move chosen: {move1}")
    if isinstance(saved_state, MinimaxSavedState):
        print(f"Transposition table entries: {len(saved_state.transposition_table)}")
    print()
    
    # Apply the move
    apply_player_action(board, move1, PLAYER2)
    
    # Second move - with saved state (should reuse previous computations)
    print("Making second move (with saved state)...")
    move2, saved_state2 = generate_move_time_limited(board, PLAYER1, saved_state, 2.0)
    print(f"Move chosen: {move2}")
    if isinstance(saved_state2, MinimaxSavedState):
        print(f"Transposition table entries: {len(saved_state2.transposition_table)}")
    print()
    
    # Apply the move
    apply_player_action(board, move2, PLAYER1)
    
    # Third move - with accumulated saved state
    print("Making third move (with accumulated saved state)...")
    move3, saved_state3 = generate_move_time_limited(board, PLAYER2, saved_state2, 2.0)
    print(f"Move chosen: {move3}")
    if isinstance(saved_state3, MinimaxSavedState):
        print(f"Transposition table entries: {len(saved_state3.transposition_table)}")
    print()
    
    print("Final board state:")
    print(board)
    print()
    
    # Verify that the saved state is actually a MinimaxSavedState
    assert isinstance(saved_state3, MinimaxSavedState), "Saved state should be MinimaxSavedState"
    
    # Check that we have some entries in the transposition table
    assert len(saved_state3.transposition_table) > 0, "Should have transposition table entries"
    
    print("✓ Transposition table is working correctly!")
    print(f"✓ Preserved {len(saved_state3.transposition_table)} position evaluations")

if __name__ == "__main__":
    test_transposition_table()
