"""
Behavioral contract tests for Connect Four heuristic evaluation.

These tests focus on the expected behavior of heuristic evaluation functions
rather than specific implementation details or hard-coded values.
"""

import numpy as np
import pytest
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2,
    check_end_state, GameState, BOARD_COLS, BOARD_ROWS, NO_PLAYER
)


class TestHeuristicBehavioralContracts:
    """Test behavioral contracts for heuristic evaluation functions."""
    
    def test_heuristic_is_symmetric_for_players(self, heuristic_func):
        """Heuristic should be symmetric - same position should give opposite scores for different players."""
        board = initialize_game_state()
        
        # Create an asymmetric position
        apply_player_action(board, PlayerAction(3), PLAYER1)
        apply_player_action(board, PlayerAction(2), PLAYER2)
        apply_player_action(board, PlayerAction(4), PLAYER1)
        
        score_p1 = heuristic_func(board, PLAYER1)
        score_p2 = heuristic_func(board, PLAYER2)
        
        # The scores should be opposite (or at least one should be better than the other)
        assert score_p1 != score_p2, "Symmetric positions should have different scores"
        
        # If one player has advantage, the other should have disadvantage
        if score_p1 > 0:
            assert score_p2 < 0, "If P1 has advantage, P2 should have disadvantage"
        elif score_p1 < 0:
            assert score_p2 > 0, "If P1 has disadvantage, P2 should have advantage"
    
    def test_heuristic_recognizes_winning_positions(self, heuristic_func):
        """Heuristic should give very high scores for winning positions."""
        board = initialize_game_state()
        
        # Create a winning position for PLAYER1
        for col in range(4):
            apply_player_action(board, PlayerAction(col), PLAYER1)
        
        score_winner = heuristic_func(board, PLAYER1)
        score_loser = heuristic_func(board, PLAYER2)
        
        # Winner should have very high score, loser should have very low score
        assert score_winner > 100, "Winning position should have high score"
        assert score_loser < -100, "Losing position should have low score"
        assert score_winner > score_loser, "Winner should have higher score than loser"
    
    def test_heuristic_prefers_center_play(self, heuristic_func):
        """Heuristic should generally prefer center columns over edge columns."""
        board_center = initialize_game_state()
        board_edge = initialize_game_state()
        
        # Place piece in center
        apply_player_action(board_center, PlayerAction(3), PLAYER1)
        
        # Place piece on edge
        apply_player_action(board_edge, PlayerAction(0), PLAYER1)
        
        score_center = heuristic_func(board_center, PLAYER1)
        score_edge = heuristic_func(board_edge, PLAYER1)
        
        # Center play should generally be preferred
        assert score_center >= score_edge, "Center play should be at least as good as edge play"
    
    def test_heuristic_responds_to_threats(self, heuristic_func):
        """Heuristic should recognize and respond to threats."""
        board_threat = initialize_game_state()
        board_no_threat = initialize_game_state()
        
        # Create a threat position (opponent has 3 in a row)
        for col in range(3):
            apply_player_action(board_threat, PlayerAction(col), PLAYER2)
        
        # Create a non-threatening position
        apply_player_action(board_no_threat, PlayerAction(0), PLAYER2)
        apply_player_action(board_no_threat, PlayerAction(2), PLAYER2)
        apply_player_action(board_no_threat, PlayerAction(4), PLAYER2)
        
        score_threat = heuristic_func(board_threat, PLAYER1)
        score_no_threat = heuristic_func(board_no_threat, PLAYER1)
        
        # Threat position should be scored worse for PLAYER1
        assert score_threat < score_no_threat, "Threat position should be scored worse"
    
    def test_heuristic_values_piece_connectivity(self, heuristic_func):
        """Heuristic should value connected pieces over scattered pieces."""
        board_connected = initialize_game_state()
        board_scattered = initialize_game_state()
        
        # Create connected pieces
        apply_player_action(board_connected, PlayerAction(3), PLAYER1)
        apply_player_action(board_connected, PlayerAction(4), PLAYER1)
        
        # Create scattered pieces
        apply_player_action(board_scattered, PlayerAction(1), PLAYER1)
        apply_player_action(board_scattered, PlayerAction(5), PLAYER1)
        
        score_connected = heuristic_func(board_connected, PLAYER1)
        score_scattered = heuristic_func(board_scattered, PLAYER1)
        
        # Connected pieces should generally be valued higher
        assert score_connected >= score_scattered, "Connected pieces should be valued at least as highly"
    
    def test_heuristic_monotonicity_with_progress(self, heuristic_func):
        """Heuristic should generally improve as player gets closer to winning."""
        boards = []
        
        # Create progression toward a win
        for num_pieces in range(1, 4):
            board = initialize_game_state()
            for col in range(num_pieces):
                apply_player_action(board, PlayerAction(col), PLAYER1)
            boards.append(board)
        
        scores = [heuristic_func(board, PLAYER1) for board in boards]
        
        # Generally, more pieces toward a win should be better
        # (allowing for some flexibility in heuristic design)
        assert scores[2] > scores[0], "3 pieces toward win should be better than 1"
    
    def test_heuristic_handles_full_board(self, heuristic_func):
        """Heuristic should handle full board positions without errors."""
        board = initialize_game_state()
        
        # Fill the board in a draw pattern
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
                board[BOARD_ROWS - 1 - row_idx, col_idx] = piece
        
        # Should not crash and should return reasonable scores
        score_p1 = heuristic_func(board, PLAYER1)
        score_p2 = heuristic_func(board, PLAYER2)
        
        assert isinstance(score_p1, (int, float)), "Should return numeric score"
        assert isinstance(score_p2, (int, float)), "Should return numeric score"
        
        # In a draw position, scores should be relatively balanced
        assert abs(score_p1 - score_p2) < 1000, "Draw position should have balanced scores"
    
    def test_heuristic_is_deterministic(self, heuristic_func):
        """Heuristic should be deterministic - same position should always give same score."""
        board = initialize_game_state()
        apply_player_action(board, PlayerAction(3), PLAYER1)
        apply_player_action(board, PlayerAction(2), PLAYER2)
        
        score1 = heuristic_func(board, PLAYER1)
        score2 = heuristic_func(board, PLAYER1)
        score3 = heuristic_func(board, PLAYER1)
        
        assert score1 == score2 == score3, "Heuristic should be deterministic"


@pytest.fixture
def heuristic_func():
    """Fixture for heuristic function."""
    from agents.agent_minimax.heuristics import evaluate_board
    return evaluate_board


def test_minimax_heuristic_contracts():
    """Test minimax heuristic behavioral contracts."""
    from agents.agent_minimax.heuristics import evaluate_board
    
    test_instance = TestHeuristicBehavioralContracts()
    
    # Run key behavioral tests
    test_instance.test_heuristic_is_symmetric_for_players(evaluate_board)
    test_instance.test_heuristic_recognizes_winning_positions(evaluate_board)
    test_instance.test_heuristic_prefers_center_play(evaluate_board)
    test_instance.test_heuristic_responds_to_threats(evaluate_board)
    test_instance.test_heuristic_is_deterministic(evaluate_board)
