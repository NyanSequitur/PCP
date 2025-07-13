"""
Unit tests for human user agent.

This module tests the human user input handling, including input validation,
conversion from user-facing to internal coordinates, and interaction flows.
"""

import pytest
import numpy as np
from agents.agent_human_user.human_user import user_move, convert_str_to_action
from game_utils import initialize_game_state, PLAYER1, PlayerAction, MoveStatus, check_move_status

def test_convert_str_to_action_valid():
    """Test that valid integer string is converted to PlayerAction (1-based user input to 0-based action)."""
    assert convert_str_to_action("3") == PlayerAction(2)

def test_convert_str_to_action_invalid():
    """Test that invalid string returns None."""
    assert convert_str_to_action("foo") is None

def test_user_move_valid(monkeypatch):
    """Test user_move returns valid PlayerAction when user inputs a valid move (1-based input)."""
    board = initialize_game_state()
    # Simulate user entering column 2 (user input 2, internal index 1)
    monkeypatch.setattr("builtins.input", lambda _: "2")
    move, _ = user_move(board, PLAYER1, None)
    assert move == PlayerAction(1)

def test_user_move_invalid_then_valid(monkeypatch):
    """Test user_move prompts again after invalid input, then accepts valid input (1-based input)."""
    board = initialize_game_state()
    # Simulate user entering 'foo' (invalid), then '1' (valid, internal index 0)
    responses = iter(["foo", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    move, _ = user_move(board, PLAYER1, None)
    assert move == PlayerAction(0)
