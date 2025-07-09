"""
Test search continuation functionality.

This module tests the search continuation feature where the minimax agent
continues from the last analyzed depth rather than starting from depth 1
when analyzing a position that was previously studied.
"""

import numpy as np
import time
import pytest

from game_utils import BoardPiece, PLAYER1, PLAYER2, apply_player_action, PlayerAction
from agents.agent_minimax.transposition_table import TranspositionTable
from agents.agent_minimax.search_continuation import iterative_deepening_search


def test_search_continuation_basic():
    """Test that search continuation works correctly."""
    # Create a test board with an interesting position
    board = np.zeros((6, 7), dtype=BoardPiece)
    
    # Set up a position with some pieces
    apply_player_action(board, PlayerAction(3), PLAYER1)
    apply_player_action(board, PlayerAction(3), PLAYER2)
    apply_player_action(board, PlayerAction(2), PLAYER1)
    apply_player_action(board, PlayerAction(4), PLAYER2)
    
    # Create transposition table
    tt = TranspositionTable()
    
    # First search - should start from depth 1
    best_col1, best_score1, completed_depth1 = iterative_deepening_search(
        board, PLAYER1, tt, time_limit_secs=0.3, max_depth=8
    )
    
    # Verify the search completed at some depth
    assert completed_depth1 > 0, "First search should complete at least depth 1"
    assert best_col1 in range(7), "Best move should be a valid column"
    
    # Check what's stored for this position
    stored_depth, stored_value, stored_best_move = tt.get_stored_result(board)
    assert stored_depth == completed_depth1, "Stored depth should match completed depth"
    assert stored_best_move == PlayerAction(best_col1), "Stored move should match best move"
    
    # Second search on the same position - should continue from where we left off
    best_col2, best_score2, completed_depth2 = iterative_deepening_search(
        board, PLAYER1, tt, time_limit_secs=0.3, max_depth=12
    )
    
    # Verify that search continuation happened
    assert completed_depth2 >= completed_depth1, "Second search should reach at least the same depth"
    assert best_col2 in range(7), "Best move should be a valid column"


def test_search_continuation_different_positions():
    """Test that different positions don't interfere with each other."""
    # Create base board
    board1 = np.zeros((6, 7), dtype=BoardPiece)
    apply_player_action(board1, PlayerAction(3), PLAYER1)
    apply_player_action(board1, PlayerAction(3), PLAYER2)
    
    # Create different board
    board2 = np.zeros((6, 7), dtype=BoardPiece)
    apply_player_action(board2, PlayerAction(2), PLAYER1)
    apply_player_action(board2, PlayerAction(4), PLAYER2)
    
    tt = TranspositionTable()
    
    # Search first position
    best_col1, best_score1, completed_depth1 = iterative_deepening_search(
        board1, PLAYER1, tt, time_limit_secs=0.2, max_depth=6
    )
    
    # Verify first position is stored
    stored_depth1, _, _ = tt.get_stored_result(board1)
    assert stored_depth1 == completed_depth1
    
    # Search second position (should start fresh)
    best_col2, best_score2, completed_depth2 = iterative_deepening_search(
        board2, PLAYER1, tt, time_limit_secs=0.2, max_depth=6
    )
    
    # Verify second position doesn't interfere with first
    stored_depth1_after, _, _ = tt.get_stored_result(board1)
    stored_depth2, _, _ = tt.get_stored_result(board2)
    
    assert stored_depth1_after == completed_depth1, "First position should remain unchanged"
    assert stored_depth2 == completed_depth2, "Second position should be stored"


def test_search_continuation_with_no_prior_analysis():
    """Test that search works normally when no prior analysis exists."""
    board = np.zeros((6, 7), dtype=BoardPiece)
    apply_player_action(board, PlayerAction(3), PLAYER1)
    
    tt = TranspositionTable()
    
    # Verify no prior analysis exists
    stored_depth, stored_value, stored_best_move = tt.get_stored_result(board)
    assert stored_depth == 0, "No prior analysis should exist"
    assert stored_value is None, "No prior value should exist"
    assert stored_best_move is None, "No prior best move should exist"
    
    # Search should work normally
    best_col, best_score, completed_depth = iterative_deepening_search(
        board, PLAYER1, tt, time_limit_secs=0.2, max_depth=5
    )
    
    assert completed_depth > 0, "Search should complete at least depth 1"
    assert best_col in range(7), "Best move should be a valid column"
    
    # Verify result is stored
    stored_depth_after, _, _ = tt.get_stored_result(board)
    assert stored_depth_after == completed_depth, "Result should be stored after search"


def test_search_continuation_performance_benefit():
    """Test that search continuation provides performance benefits."""
    # Create a moderately complex position
    board = np.zeros((6, 7), dtype=BoardPiece)
    apply_player_action(board, PlayerAction(3), PLAYER1)
    apply_player_action(board, PlayerAction(3), PLAYER2)
    apply_player_action(board, PlayerAction(2), PLAYER1)
    apply_player_action(board, PlayerAction(4), PLAYER2)
    apply_player_action(board, PlayerAction(1), PLAYER1)
    
    tt = TranspositionTable()
    
    # First search to populate transposition table
    start_time = time.monotonic()
    best_col1, _, completed_depth1 = iterative_deepening_search(
        board, PLAYER1, tt, time_limit_secs=0.4, max_depth=8
    )
    first_search_time = time.monotonic() - start_time
    
    # Second search should be faster due to continuation
    start_time = time.monotonic()
    best_col2, _, completed_depth2 = iterative_deepening_search(
        board, PLAYER1, tt, time_limit_secs=0.4, max_depth=10
    )
    second_search_time = time.monotonic() - start_time
    
    # Verify search continuation happened
    assert completed_depth2 >= completed_depth1, "Second search should reach deeper or same depth"
    
    # The second search should either be faster or reach greater depth
    # (allowing some tolerance for timing variations)
    depth_improvement = completed_depth2 - completed_depth1
    time_ratio = second_search_time / max(first_search_time, 0.001)  # Prevent division by zero
    
    # Either we got deeper analysis or it was significantly faster
    assert depth_improvement > 0 or time_ratio < 0.8, \
        f"Search continuation should provide benefit: depth improvement={depth_improvement}, time ratio={time_ratio:.2f}"


def test_search_continuation_game_scenario():
    """Test search continuation in a realistic game scenario."""
    # Create a moderately complex position
    board = np.zeros((6, 7), dtype=BoardPiece)
    
    # Set up a mid-game position with some strategic complexity
    apply_player_action(board, PlayerAction(3), PLAYER1)  # Center column
    apply_player_action(board, PlayerAction(3), PLAYER2)  # Block center
    apply_player_action(board, PlayerAction(2), PLAYER1)  # Left of center
    apply_player_action(board, PlayerAction(4), PLAYER2)  # Right of center
    apply_player_action(board, PlayerAction(1), PLAYER1)  # Further left
    apply_player_action(board, PlayerAction(5), PLAYER2)  # Further right
    apply_player_action(board, PlayerAction(3), PLAYER1)  # Stack on center
    
    tt = TranspositionTable()
    
    # Agent analyzes the position and finds the best move
    agent_best_col, agent_score, agent_depth = iterative_deepening_search(
        board, PLAYER1, tt, time_limit_secs=0.5, max_depth=10
    )
    
    print(f"Agent analysis: move={agent_best_col}, score={agent_score:.1f}, depth={agent_depth}")
    
    # Verify agent found a valid move and analyzed to some depth
    assert agent_best_col in range(7), "Agent should find a valid move"
    assert agent_depth > 0, "Agent should complete some analysis"
    
    # Agent plays the best move
    board_after_agent = board.copy()
    apply_player_action(board_after_agent, PlayerAction(agent_best_col), PLAYER1)
    
    # Dummy player (PLAYER2) analyzes and plays the best response
    dummy_best_col, dummy_score, dummy_depth = iterative_deepening_search(
        board_after_agent, PLAYER2, tt, time_limit_secs=0.3, max_depth=8
    )
    
    print(f"Dummy response: move={dummy_best_col}, score={dummy_score:.1f}, depth={dummy_depth}")
    
    # Dummy player plays their best response
    board_after_dummy = board_after_agent.copy()
    apply_player_action(board_after_dummy, PlayerAction(dummy_best_col), PLAYER2)
    
    # Check if agent has any prior analysis for this new position
    prior_depth, prior_value, prior_move = tt.get_stored_result(board_after_dummy)
    print(f"Prior analysis for new position: depth={prior_depth}, value={prior_value}, move={prior_move}")
    
    # Agent analyzes the new position after dummy's move
    # This should benefit from search continuation if the position was predicted
    agent_second_col, agent_second_score, agent_second_depth = iterative_deepening_search(
        board_after_dummy, PLAYER1, tt, time_limit_secs=0.5, max_depth=12
    )
    
    print(f"Agent second analysis: move={agent_second_col}, score={agent_second_score:.1f}, depth={agent_second_depth}")
    
    # Verify the agent found a valid move in the new position
    assert agent_second_col in range(7), "Agent should find a valid move in new position"
    assert agent_second_depth > 0, "Agent should complete analysis of new position"
    
    # The key test: if there was prior analysis, the agent should start from a deeper depth
    # We can verify this by checking that the transposition table has entries for the position
    # and that the search was efficient (either deeper analysis or faster completion)
    if prior_depth > 0:
        print(f"✓ Search continuation detected: prior depth was {prior_depth}")
        # The new analysis should reach at least the prior depth
        assert agent_second_depth >= prior_depth, \
            f"With continuation, should reach at least prior depth {prior_depth}, got {agent_second_depth}"
    else:
        print("ℹ No prior analysis found - position was not predicted")
    
    # Verify the game state is consistent
    initial_pieces = np.count_nonzero(board)
    final_pieces = np.count_nonzero(board_after_dummy)
    assert final_pieces == initial_pieces + 2, f"Should have exactly 2 more pieces after both moves: {initial_pieces} -> {final_pieces}"
    
    print("Game scenario test completed successfully!")


if __name__ == "__main__":
    # Run the tests in verbose mode if executed directly
    pytest.main([__file__, "-v"])
