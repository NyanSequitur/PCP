"""
Unit tests for the minimax heuristic evaluation functions.

This module tests the heuristic evaluation functions used by the minimax agent
to score board positions.
"""

import numpy as np
import pytest
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2, 
    NO_PLAYER, BOARD_COLS, BOARD_ROWS
)
from agents.agent_minimax.heuristics import (
    evaluate_board, get_other_player, _score_window, _are_pieces_connected,
    _score_horizontal_windows, _score_vertical_windows, _score_diagonal_windows,
    _score_center_column, _score_positional_weights, _detect_multiple_threats,
    _score_connectivity
)


class TestHeuristicUtilities:
    """Test utility functions for heuristics."""
    
    def test_get_other_player(self):
        """Test getting the other player."""
        assert get_other_player(PLAYER1) == PLAYER2
        assert get_other_player(PLAYER2) == PLAYER1
    
    def test_are_pieces_connected(self):
        """Test piece connectivity detection."""
        # Test connected pieces
        window = np.array([PLAYER1, PLAYER1, NO_PLAYER, NO_PLAYER])
        assert _are_pieces_connected(window, PLAYER1)
        
        # Test disconnected pieces
        window = np.array([PLAYER1, NO_PLAYER, PLAYER1, NO_PLAYER])
        assert not _are_pieces_connected(window, PLAYER1)
        
        # Test single piece
        window = np.array([PLAYER1, NO_PLAYER, NO_PLAYER, NO_PLAYER])
        assert _are_pieces_connected(window, PLAYER1)
        
        # Test no pieces - implementation considers no pieces as "trivially connected"
        window = np.array([NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER])
        assert _are_pieces_connected(window, PLAYER1)


class TestWindowScoring:
    """Test scoring of 4-cell windows."""
    
    def test_score_window_win(self):
        """Test window scoring for winning positions."""
        window = np.array([PLAYER1, PLAYER1, PLAYER1, PLAYER1])
        score = _score_window(window, PLAYER1)
        assert score == 1000.0
    
    def test_score_window_loss(self):
        """Test window scoring for losing positions."""
        window = np.array([PLAYER2, PLAYER2, PLAYER2, PLAYER2])
        score = _score_window(window, PLAYER1)
        assert score == -1000.0
    
    def test_score_window_three_in_a_row(self):
        """Test window scoring for three pieces in a row."""
        window = np.array([PLAYER1, PLAYER1, PLAYER1, NO_PLAYER])
        score = _score_window(window, PLAYER1)
        assert score > 0
        assert score < 1000.0
    
    def test_score_window_opponent_three_in_a_row(self):
        """Test window scoring for opponent three in a row."""
        window = np.array([PLAYER2, PLAYER2, PLAYER2, NO_PLAYER])
        score = _score_window(window, PLAYER1)
        assert score < 0
        assert score > -1000.0
    
    def test_score_window_two_in_a_row(self):
        """Test window scoring for two pieces in a row."""
        window = np.array([PLAYER1, PLAYER1, NO_PLAYER, NO_PLAYER])
        score = _score_window(window, PLAYER1)
        assert score > 0
        assert score < 50.0
    
    def test_score_window_blocked(self):
        """Test window scoring for blocked positions."""
        window = np.array([PLAYER1, PLAYER1, PLAYER2, NO_PLAYER])
        score = _score_window(window, PLAYER1)
        assert score == 0.0
    
    def test_score_window_empty(self):
        """Test window scoring for empty window."""
        window = np.array([NO_PLAYER, NO_PLAYER, NO_PLAYER, NO_PLAYER])
        score = _score_window(window, PLAYER1)
        assert score == 0.0


class TestDirectionalScoring:
    """Test directional scoring functions."""
    
    def test_score_horizontal_windows(self):
        """Test horizontal window scoring."""
        board = initialize_game_state()
        
        # Empty board should have neutral score
        score = _score_horizontal_windows(board, PLAYER1)
        assert score >= 0
        
        # Add some pieces horizontally
        for i in range(3):
            apply_player_action(board, PlayerAction(i), PLAYER1)
        
        score = _score_horizontal_windows(board, PLAYER1)
        assert score > 0
    
    def test_score_vertical_windows(self):
        """Test vertical window scoring."""
        board = initialize_game_state()
        
        # Empty board should have neutral score
        score = _score_vertical_windows(board, PLAYER1)
        assert score >= 0
        
        # Add some pieces vertically
        for _ in range(3):
            apply_player_action(board, PlayerAction(3), PLAYER1)
        
        score = _score_vertical_windows(board, PLAYER1)
        assert score > 0
    
    def test_score_diagonal_windows(self):
        """Test diagonal window scoring."""
        board = initialize_game_state()
        
        # Empty board should have neutral score
        score = _score_diagonal_windows(board, PLAYER1)
        assert score >= 0
        
        # Create a diagonal setup
        apply_player_action(board, PlayerAction(0), PLAYER1)
        apply_player_action(board, PlayerAction(1), PLAYER2)
        apply_player_action(board, PlayerAction(1), PLAYER1)
        apply_player_action(board, PlayerAction(2), PLAYER2)
        apply_player_action(board, PlayerAction(2), PLAYER2)
        apply_player_action(board, PlayerAction(2), PLAYER1)
        
        score = _score_diagonal_windows(board, PLAYER1)
        assert score > 0


class TestPositionalScoring:
    """Test positional scoring functions."""
    
    def test_score_center_column(self):
        """Test center column scoring."""
        board = initialize_game_state()
        
        # Empty board should have neutral score
        score = _score_center_column(board, PLAYER1)
        assert score == 0
        
        # Add piece to center column (bottom row gets highest weight)
        apply_player_action(board, PlayerAction(3), PLAYER1)
        score = _score_center_column(board, PLAYER1)
        expected_score = 4.0 * BOARD_ROWS  # 4.0 * 6 = 24.0
        assert score == expected_score
        
        # Add opponent piece to center column
        apply_player_action(board, PlayerAction(3), PLAYER2)
        score = _score_center_column(board, PLAYER1)
        # Player1 gets +24.0, Player2 gets -3.0 * 5 = -15.0, total = 9.0
        expected_score = 24.0 - 15.0  # 9.0
        assert score == expected_score
    
    def test_score_positional_weights(self):
        """Test positional weight scoring."""
        board = initialize_game_state()
        
        # Empty board should have neutral score
        score = _score_positional_weights(board, PLAYER1)
        assert score == 0
        
        # Add piece to center (should be weighted higher)
        apply_player_action(board, PlayerAction(3), PLAYER1)
        center_score = _score_positional_weights(board, PLAYER1)
        
        # Reset and add piece to edge
        board = initialize_game_state()
        apply_player_action(board, PlayerAction(0), PLAYER1)
        edge_score = _score_positional_weights(board, PLAYER1)
        
        # Center should be weighted higher
        assert center_score > edge_score


class TestAdvancedHeuristics:
    """Test advanced heuristic functions."""
    
    def test_detect_multiple_threats(self):
        """Test multiple threat detection."""
        board = initialize_game_state()
        
        # Empty board should have no threats
        score = _detect_multiple_threats(board, PLAYER1)
        assert score == 0
        
        # Create a position with potential threats
        apply_player_action(board, PlayerAction(1), PLAYER1)
        apply_player_action(board, PlayerAction(2), PLAYER1)
        apply_player_action(board, PlayerAction(4), PLAYER1)
        apply_player_action(board, PlayerAction(5), PLAYER1)
        
        score = _detect_multiple_threats(board, PLAYER1)
        assert score >= 0
    
    def test_score_connectivity(self):
        """Test connectivity scoring."""
        board = initialize_game_state()
        
        # Empty board should have neutral score
        score = _score_connectivity(board, PLAYER1)
        assert score == 0
        
        # Add connected pieces
        apply_player_action(board, PlayerAction(3), PLAYER1)
        apply_player_action(board, PlayerAction(4), PLAYER1)
        
        score = _score_connectivity(board, PLAYER1)
        assert score > 0
        
        # Add disconnected piece
        apply_player_action(board, PlayerAction(0), PLAYER1)
        new_score = _score_connectivity(board, PLAYER1)
        assert new_score >= score  # Should not decrease


class TestBoardEvaluation:
    """Test the main board evaluation function."""
    
    def test_evaluate_board_empty(self):
        """Test evaluation of empty board."""
        board = initialize_game_state()
        
        score = evaluate_board(board, PLAYER1)
        assert isinstance(score, float)
        assert score >= 0  # Should be neutral/positive for empty board
    
    def test_evaluate_board_winning_position(self):
        """Test evaluation of winning position."""
        board = initialize_game_state()
        
        # Create a winning position for PLAYER1
        for i in range(4):
            apply_player_action(board, PlayerAction(i), PLAYER1)
        
        score = evaluate_board(board, PLAYER1)
        assert score > 500  # Should be very positive
    
    def test_evaluate_board_losing_position(self):
        """Test evaluation of losing position."""
        board = initialize_game_state()
        
        # Create a winning position for PLAYER2 (losing for PLAYER1)
        for i in range(4):
            apply_player_action(board, PlayerAction(i), PLAYER2)
        
        score = evaluate_board(board, PLAYER1)
        assert score < -500  # Should be very negative
    
    def test_evaluate_board_symmetry(self):
        """Test that symmetric positions have similar evaluations."""
        board1 = initialize_game_state()
        board2 = initialize_game_state()
        
        # Create symmetric positions
        apply_player_action(board1, PlayerAction(1), PLAYER1)
        apply_player_action(board1, PlayerAction(2), PLAYER2)
        
        apply_player_action(board2, PlayerAction(5), PLAYER1)  # Mirror of 1
        apply_player_action(board2, PlayerAction(4), PLAYER2)  # Mirror of 2
        
        score1 = evaluate_board(board1, PLAYER1)
        score2 = evaluate_board(board2, PLAYER1)
        
        # Scores should be close (within 10% tolerance)
        assert abs(score1 - score2) < abs(score1) * 0.1
    
    def test_evaluate_board_player_perspective(self):
        """Test that evaluation depends on player perspective."""
        board = initialize_game_state()
        
        # Create a position favorable to PLAYER1
        for i in range(3):
            apply_player_action(board, PlayerAction(i), PLAYER1)
        
        score_p1 = evaluate_board(board, PLAYER1)
        score_p2 = evaluate_board(board, PLAYER2)
        
        # Should be positive for PLAYER1, negative for PLAYER2
        assert score_p1 > 0
        assert score_p2 < 0
    
    def test_evaluate_board_progressive_improvement(self):
        """Test that better positions get progressively better scores."""
        board = initialize_game_state()
        
        # Empty board score
        score0 = evaluate_board(board, PLAYER1)
        
        # One piece
        apply_player_action(board, PlayerAction(3), PLAYER1)
        score1 = evaluate_board(board, PLAYER1)
        
        # Two pieces
        apply_player_action(board, PlayerAction(2), PLAYER1)
        score2 = evaluate_board(board, PLAYER1)
        
        # Three pieces (threatening)
        apply_player_action(board, PlayerAction(1), PLAYER1)
        score3 = evaluate_board(board, PLAYER1)
        
        # Scores should generally increase
        assert score1 >= score0
        assert score2 >= score1
        assert score3 > score2  # Three in a row should be significantly better
    
    def test_evaluate_board_consistency(self):
        """Test that evaluation is consistent for the same position."""
        board = initialize_game_state()
        
        # Add some pieces
        apply_player_action(board, PlayerAction(3), PLAYER1)
        apply_player_action(board, PlayerAction(4), PLAYER2)
        apply_player_action(board, PlayerAction(2), PLAYER1)
        
        # Evaluate multiple times
        scores = [evaluate_board(board, PLAYER1) for _ in range(5)]
        
        # All scores should be identical
        assert all(score == scores[0] for score in scores)
    
    def test_evaluate_board_reasonable_range(self):
        """Test that evaluation scores are in reasonable ranges."""
        board = initialize_game_state()
        
        # Test various positions
        positions = [
            [],  # Empty
            [3],  # Center
            [0, 1, 2],  # Edge sequence
            [3, 4, 5, 6],  # Center sequence
            [0, 6, 1, 5, 2, 4],  # Alternating
        ]
        
        for pos in positions:
            test_board = initialize_game_state()
            for i, col in enumerate(pos):
                player = PLAYER1 if i % 2 == 0 else PLAYER2
                apply_player_action(test_board, PlayerAction(col), player)
            
            score = evaluate_board(test_board, PLAYER1)
            
            # Score should be within reasonable bounds (not infinite)
            assert -10000 < score < 10000
            assert isinstance(score, float)
