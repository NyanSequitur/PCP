"""
Human user agent for Connect Four.

Provides functions to query the user for a move and convert input to a valid PlayerAction.
"""

import numpy as np
from typing import Callable, Any

from game_utils import BoardPiece, PlayerAction, SavedState, MoveStatus, check_move_status, user_col_to_internal


def query_user(prompt_function: Callable[[str], str]) -> str:
    """
    Prompt the user for input using the provided prompt function.

    Parameters
    ----------
    prompt_function : Callable[[str], str]
        Function to prompt the user (e.g., input).

    Returns
    -------
    str
        The user's input as a string.
    """
    return prompt_function("Column? ")


def user_move(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    """
    Query the user for a move and validate it.

    Parameters
    ----------
    board : np.ndarray
        The current game board.
    player : BoardPiece
        The player making the move.
    saved_state : SavedState | None
        State to persist between moves.

    Returns
    -------
    tuple[PlayerAction, SavedState | None]
        The chosen move and updated state.
    """
    move_status: MoveStatus | None = None
    input_move: PlayerAction | None = None
    while move_status != MoveStatus.IS_VALID:
        print(f"\nYour turn! You are playing as: {'X' if player == 1 else 'O'}")
        print("Enter the column number (1-7) where you want to drop your piece.")
        input_move_string = query_user(input)
        input_move = convert_str_to_action(input_move_string)
        if input_move is None:
            continue
        move_status = check_move_status(board, input_move)
        if move_status != MoveStatus.IS_VALID:
            print(f"Move is invalid: {move_status.value}")
            print("Try again.")
    # input_move is guaranteed to be PlayerAction here
    assert input_move is not None
    return input_move, saved_state


def convert_str_to_action(input_move_string: str) -> PlayerAction | None:
    """
    Convert a string to a PlayerAction if possible, using 1-based user input.

    Parameters
    ----------
    input_move_string : str
        The user's input string.

    Returns
    -------
    PlayerAction | None
        The corresponding PlayerAction, or None if invalid.
    """
    try:
        user_col = int(input_move_string)
        if not 1 <= user_col <= 7:
            print("Invalid move: Input must be an integer between 1 and 7.")
            print("Try again.")
            return None
        internal_col = user_col_to_internal(user_col)
        return PlayerAction(internal_col)
    except ValueError:
        print("Invalid move: Input must be an integer between 1 and 7.")
        print("Try again.")
        return None
