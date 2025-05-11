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
    board = initialize_game_state()
    assert board.shape == BOARD_SHAPE
    assert board.dtype == BoardPiece
    assert np.all(board == NO_PLAYER)


def test_pretty_print_roundtrip():
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[1, 0] = PLAYER2
    board[0, 3] = PLAYER1

    printed = pretty_print_board(board)
    restored = string_to_board(printed)

    assert np.array_equal(board, restored)


# ──────────────────── apply_player_action ────────────────────


def test_apply_player_action_stacks_from_bottom():
    board = initialize_game_state()

    apply_player_action(board, PlayerAction(0), PLAYER1)
    apply_player_action(board, PlayerAction(0), PLAYER2)

    assert board[0, 0] == PLAYER1
    assert board[1, 0] == PLAYER2


def test_apply_player_action_full_column_raises():
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
    board = initialize_game_state()
    for r, c in coords:
        board[r, c] = PLAYER1

    assert connected_four(board, PLAYER1)
    assert not connected_four(board, PLAYER2)


# ──────────────────── game state ────────────────────


def test_check_end_state_win():
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
