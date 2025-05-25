"""
Main entry point for Connect Four game.

Provides a function to play a game between two agents (human or AI).
"""

from typing import Callable
import time

from game_utils import (
    PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT,
    GameState, MoveStatus, GenMove,
    initialize_game_state, pretty_print_board,
    apply_player_action, check_end_state, check_move_status
)
from agents.agent_human_user import user_move


def human_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove = user_move,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None,
):
    """
    Play a game between two agents (human or AI).

    Parameters
    ----------
    generate_move_1 : GenMove
        Move generator for player 1.
    generate_move_2 : GenMove, optional
        Move generator for player 2 (default is user_move).
    player_1 : str, optional
        Name for player 1 (default is "Player 1").
    player_2 : str, optional
        Name for player 2 (default is "Player 2").
    args_1 : tuple, optional
        Extra arguments for player 1's move generator.
    args_2 : tuple, optional
        Extra arguments for player 2's move generator.
    init_1 : Callable, optional
        Initialization function for player 1.
    init_2 : Callable, optional
        Initialization function for player 2.
    """
    players = (PLAYER1, PLAYER2)
    # Play each game twice, swapping who goes first each time
    for play_first in (1, -1):
        # Initialize both agents (if needed) before each game
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state: dict = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        # Swap move generators and player names if play_first == -1
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with '
                    f'{PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                )
                # Pass a copy of the board to prevent agents from modifying the real board
                action, saved_state[player] = gen_move(
                    board.copy(),
                    player, saved_state[player], *args
                )

                move_status = check_move_status(board, action)
                if move_status != MoveStatus.IS_VALID:
                    print(f'Move {action} is invalid: {move_status.value}')
                    print(f'{player_name} lost by making an illegal move.')
                    playing = False
                    break

                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print('Game ended in draw')
                    else:
                        # Fanfare for the winner
                        print("\n" + "*" * 40)
                        print("  🎉🎉🎉  CONGRATULATIONS!  🎉🎉🎉  ")
                        print(f"  {player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}")
                        print("*" * 40 + "\n")
                        import time
                        input("Press Enter to continue...")
                    playing = False
                    break



if __name__ == "__main__":
    #human_vs_agent(user_move)
    from agents.agent_minimax import generate_move as minimax_move

    human_vs_agent(user_move, minimax_move, args_2=())