import numpy as np
from game_utils import initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2
from agents.agent_minimax.minimax import MinimaxSavedState

# Debug the symmetry issue
state = MinimaxSavedState()

# Create a board and its mirror
board = initialize_game_state()
apply_player_action(board, PlayerAction(1), PLAYER1)  # Left side
apply_player_action(board, PlayerAction(2), PLAYER2)

mirrored_board = initialize_game_state()
apply_player_action(mirrored_board, PlayerAction(5), PLAYER1)  # Right side (mirrored)
apply_player_action(mirrored_board, PlayerAction(4), PLAYER2)

print("Board 1:")
print(board)
print("\nBoard 2:")
print(mirrored_board)
print("\nBoard 1 mirrored:")
print(np.fliplr(board))
print("\nAre they equal?", np.array_equal(np.fliplr(board), mirrored_board))

# Test canonical form
canonical1 = state._get_canonical_board(board)
canonical2 = state._get_canonical_board(mirrored_board)
print("\nCanonical 1:")
print(canonical1)
print("\nCanonical 2:")
print(canonical2)
print("\nCanonical forms equal?", np.array_equal(canonical1, canonical2))
