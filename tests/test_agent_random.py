"""
Unit tests for random agent.

This module tests the random agent's move generation, ensuring it produces
valid moves and behaves correctly under various board conditions.
"""

import numpy as np
from agents.agent_random.random import generate_move_random
from game_utils import initialize_game_state, PLAYER1, BOARD_COLS, PlayerAction

def test_random_agent_uniform_distribution():
    """Test that the random agent selects all columns with roughly equal probability on an empty board."""
    board = initialize_game_state()
    counts = np.zeros(BOARD_COLS, dtype=int)
    num_trials = 10000
    for _ in range(num_trials):
        move, _ = generate_move_random(board, PLAYER1, None)
        counts[int(move)] += 1
    # All columns should be chosen, and the min/max should not differ by more than 20% (loose check)
    assert np.all(counts > 0), f"Some columns were never chosen: {counts}"
    min_count = counts.min()
    max_count = counts.max()
    assert min_count > 0.8 * (num_trials / BOARD_COLS), f"Distribution too uneven: {counts}"
    assert max_count < 1.2 * (num_trials / BOARD_COLS), f"Distribution too uneven: {counts}"
