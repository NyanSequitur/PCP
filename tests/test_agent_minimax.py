"""
Unit tests for the Minimax agent.
"""
import numpy as np
import pytest
import time
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2, GameState, check_end_state
)
from agents.agent_minimax.minimax import generate_move_time_limited

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
