"""
Main entry point for Connect Four game.

Provides a function to play a game between two agents (human or AI).
"""

from typing import Callable
import time
import importlib
import sys
import os

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
    Play a single game between two agents (human or AI).

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
    # Only play one game per call
    for init, player in zip((init_1, init_2), players):
        init(initialize_game_state(), player)

    saved_state: dict = {PLAYER1: None, PLAYER2: None}
    board = initialize_game_state()
    gen_moves = (generate_move_1, generate_move_2)
    player_names = (player_1, player_2)
    gen_args = (args_1, args_2)

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
                    input("Press Enter to continue...")
                playing = False
                break

def list_available_agents():
    """
    Get a list of all available agents for the game.

    Returns
    -------
    list[dict]
        List of agent information dictionaries, each containing:
        - name: Internal agent name
        - module: Python module path
        - func: Function name to call
        - display: Human-readable display name
        - args: Default arguments tuple
    """
    agent_infos = [
        {
            'name': 'Human',
            'module': 'agents.agent_human_user',
            'func': 'user_move',
            'display': 'Human User',
            'args': (),
        },
        {
            'name': 'Random',
            'module': 'agents.agent_random',
            'func': 'generate_move',
            'display': 'Random AI',
            'args': (),
        },
        {
            'name': 'Minimax',
            'module': 'agents.agent_minimax',
            'func': 'generate_move',
            'display': 'Minimax AI',
            'args': (),
        },
    ]
    return agent_infos

def pick_agent(player_label):
    """
    Interactive agent selection for a player.

    Displays available agents and prompts the user to select one.
    Continues prompting until a valid selection is made.

    Parameters
    ----------
    player_label : str
        Label for the player (e.g., "Player 1 (X)") to display in prompts.

    Returns
    -------
    tuple[Callable, str, tuple]
        A tuple containing:
        - The agent's move generation function
        - The agent's display name
        - The agent's default arguments
    """
    agents = list_available_agents()
    print(f"\nSelect agent for {player_label}:")
    for idx, agent in enumerate(agents, 1):
        print(f"  {idx}. {agent['display']}")
    while True:
        choice = input(f"Enter number (1-{len(agents)}): ")
        try:
            idx = int(choice)
            if 1 <= idx <= len(agents):
                agent = agents[idx-1]
                module = importlib.import_module(agent['module'])
                func = getattr(module, agent['func'])
                return func, agent['display'], agent['args']
        except Exception:
            pass
        print("Invalid choice. Try again.")

def play_game_with_agent_selection():
    """
    Main game loop with interactive agent selection.

    Allows users to:
    - Select agents for both players
    - Play multiple games with the same agents
    - Swap sides between games
    - Change agents or quit

    The function continues until the user chooses to quit.
    """
    while True:
        func1, name1, args1 = pick_agent("Player 1 (X)")
        func2, name2, args2 = pick_agent("Player 2 (O)")
        swap = False
        while True:
            if not swap:
                f1, n1, a1 = func1, name1, args1
                f2, n2, a2 = func2, name2, args2
            else:
                f1, n1, a1 = func2, name2, args2
                f2, n2, a2 = func1, name1, args1
            human_vs_agent(
                f1, f2,
                player_1=n1, player_2=n2,
                args_1=a1, args_2=a2
            )
            print("\nGame over.")
            again = input("Play again with same agents (swap sides)? (y = yes, n = change agents, q = quit): ").strip().lower()
            if again == 'y':
                swap = not swap
                continue
            elif again == 'n':
                break
            elif again == 'q':
                print("Goodbye!")
                sys.exit(0)
            else:
                print("Invalid input. Please enter 'y', 'n', or 'q'.")

if __name__ == "__main__":
    play_game_with_agent_selection()