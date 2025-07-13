"""
Unit tests for game_utils module.

This module tests the core game logic including board initialization,
move validation, game state checking, and utility functions.
"""

import pytest
import numpy as np

from game_utils import (
    BOARD_ROWS,
    BOARD_COLS,
    BOARD_SHAPE,
    BoardPiece,
    NO_PLAYER,
    PLAYER1,
    PLAYER2,
    PlayerAction,
    GameState,
    MoveStatus,
    initialize_game_state,
    pretty_print_board,
    string_to_board,
    apply_player_action,
    connected_four,
    check_end_state,
    check_move_status,
)


# ──────────────────── basic board helpers ────────────────────


def test_initialize_game_state_shape_dtype():
    """Test that the initialized board has correct shape, dtype, and is empty."""
    board = initialize_game_state()
    assert board.shape == BOARD_SHAPE
    assert board.dtype == BoardPiece
    assert np.all(board == NO_PLAYER)


def test_pretty_print_board_format():
    """Test that pretty_print_board produces the expected format."""
    board = initialize_game_state()
    board[0, 0] = PLAYER1  # Bottom-left
    board[1, 0] = PLAYER2  # One row up from bottom-left
    board[0, 3] = PLAYER1  # Bottom, column 4

    printed = pretty_print_board(board)
    lines = printed.splitlines()
    
    # Check basic structure
    assert len(lines) == BOARD_ROWS + 6  # board rows + legend + borders + index
    
    # Check legend is present
    assert "Legend:" in lines[1]
    assert "X = Player 1" in lines[1]
    assert "O = Player 2" in lines[1]
    
    # Check borders
    horizontal_border = f"|{'=' * (BOARD_COLS * 3)}|"
    assert lines[3] == horizontal_border
    assert lines[-2] == horizontal_border
    
    # Check column indices at bottom
    expected_index_line = "| 1  2  3  4  5  6  7 |"
    assert lines[-1] == expected_index_line
    
    # Check that pieces appear in correct positions
    # The bottom row (board[0, :]) should be the second-to-last board row in the output
    bottom_row_line = lines[-3]  # Second from bottom, before border
    assert " X " in bottom_row_line  # PLAYER1 at position 0
    assert " X " in bottom_row_line  # PLAYER1 at position 3
    
    # Check row numbers are displayed
    assert "← 1" in bottom_row_line  # Bottom row should show row 1


def test_string_to_board_parsing():
    """Test that string_to_board correctly parses a board string."""
    # Create a known board state
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[1, 0] = PLAYER2
    board[0, 3] = PLAYER1
    board[5, 6] = PLAYER2  # Top-right corner
    
    # Convert to string and back
    printed = pretty_print_board(board)
    restored = string_to_board(printed)
    
    # Check specific positions
    assert restored[0, 0] == PLAYER1
    assert restored[1, 0] == PLAYER2
    assert restored[0, 3] == PLAYER1
    assert restored[5, 6] == PLAYER2
    
    # Check that empty positions remain empty
    assert restored[0, 1] == NO_PLAYER
    assert restored[2, 2] == NO_PLAYER


def test_pretty_print_roundtrip():
    """Test that pretty_print_board and string_to_board are inverses."""
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[1, 0] = PLAYER2
    board[0, 3] = PLAYER1

    printed = pretty_print_board(board)
    restored = string_to_board(printed)

    assert np.array_equal(board, restored)


# ──────────────────── apply_player_action ────────────────────


def test_apply_player_action_stacks_from_bottom():
    """Test that pieces stack from the bottom of the column."""
    board = initialize_game_state()

    apply_player_action(board, PlayerAction(0), PLAYER1)
    apply_player_action(board, PlayerAction(0), PLAYER2)

    assert board[0, 0] == PLAYER1
    assert board[1, 0] == PLAYER2


def test_apply_player_action_full_column_raises():
    """Test that applying to a full column raises ValueError."""
    board = initialize_game_state()
    for _ in range(BOARD_ROWS):
        apply_player_action(board, PlayerAction(1), PLAYER1)

    with pytest.raises(ValueError):
        apply_player_action(board, PlayerAction(1), PLAYER2)


# ──────────────────── connected_four ────────────────────


@pytest.mark.parametrize(
    "coords",
    [
        # horizontal
        [(0, c) for c in range(4)],
        # vertical
        [(r, 0) for r in range(4)],
        # diagonal ↗︎
        [(i, i) for i in range(4)],
        # diagonal ↘︎
        [(i, 3 - i) for i in range(4)],
    ],
    ids=["horizontal", "vertical", "diag_up", "diag_down"],
)
def test_connected_four_detects_win(coords):
    """Test that connected_four detects a win for PLAYER1 and not for PLAYER2."""
    board = initialize_game_state()
    for r, c in coords:
        board[r, c] = PLAYER1

    assert connected_four(board, PLAYER1)
    assert not connected_four(board, PLAYER2)


# ──────────────────── game state ────────────────────


def test_check_end_state_win():
    """Test that check_end_state detects a win in all directions."""
    board = initialize_game_state()

    # horizontal
    for c in range(4):
        board[0, c] = PLAYER1
    assert check_end_state(board, PLAYER1) is GameState.IS_WIN

    # vertical
    for r in range(4):
        board[r, 0] = PLAYER1
    assert check_end_state(board, PLAYER1) is GameState.IS_WIN

    # diagonal ↗︎
    for i in range(4):
        board[i, i] = PLAYER1
    assert check_end_state(board, PLAYER1) is GameState.IS_WIN

    # diagonal ↘︎
    for i in range(4):
        board[i, 3 - i] = PLAYER1
    assert check_end_state(board, PLAYER1) is GameState.IS_WIN


def test_check_end_state_draw():
    """Test that check_end_state detects a draw for both players."""
    board = initialize_game_state()

    # Fill the board with a known draw pattern (alternating X and O)
    draw_pattern = [
        [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],  # Top row
        [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
        [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
        [PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2],
        [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],
        [PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1, PLAYER2, PLAYER1],  # Bottom row
    ]

    for row_idx, row in enumerate(draw_pattern):
        for col_idx, piece in enumerate(row):
            board[BOARD_ROWS - 1 - row_idx, col_idx] = piece

    assert check_end_state(board, PLAYER1) == GameState.IS_DRAW
    assert check_end_state(board, PLAYER2) == GameState.IS_DRAW


# ──────────────────── move validation ────────────────────


def test_check_move_status_conditions():
    """Test all move status conditions: type, bounds, full column, valid."""
    board = initialize_game_state()

    # type validation
    assert check_move_status(board, 3) is MoveStatus.WRONG_TYPE

    # bounds
    assert check_move_status(board, PlayerAction(-1)) is MoveStatus.OUT_OF_BOUNDS
    assert check_move_status(board, PlayerAction(BOARD_COLS)) is MoveStatus.OUT_OF_BOUNDS

    # full column
    for _ in range(BOARD_ROWS):
        apply_player_action(board, PlayerAction(2), PLAYER1)
    assert check_move_status(board, PlayerAction(2)) is MoveStatus.FULL_COLUMN

    # valid
    assert check_move_status(board, PlayerAction(3)) is MoveStatus.IS_VALID
