"""
Benchmark module for testing minimax agent win rates against random player.

This module provides functions to test the performance of the minimax agent
by playing multiple games against a random agent and calculating win rates.
"""

import sys
import copy
from game_utils import (
    PLAYER1, PLAYER2, GameState,
    initialize_game_state,
    apply_player_action, check_end_state, check_move_status
)
from agents.agent_minimax import minimax
from agents.agent_minimax.saved_state import MinimaxSavedState
from agents.agent_random import random as random_agent

def get_winner(board, player):
    """
    Determine the winner of a completed game.

    Parameters
    ----------
    board : np.ndarray
        The game board to check.
    player : BoardPiece
        The player who just made a move.

    Returns
    -------
    int
        1 if player won, -1 if player lost, 0 if draw, None if game continues.
    """
    result = check_end_state(board, player)
    if result == GameState.IS_DRAW:
        return 0
    elif result == GameState.IS_WIN:
        return player
    elif result == GameState.IS_LOSS:
        return -player
    return None

def play_minimax_vs_random(minimax_first=True, max_depth=6):
    """
    Play a single game between minimax and random agents.

    Parameters
    ----------
    minimax_first : bool, optional
        If True, minimax agent plays first (as PLAYER1). Default is True.
    max_depth : int, optional
        Maximum search depth for minimax agent. Default is 6.

    Returns
    -------
    int
        Winner of the game: 1 for PLAYER1, -1 for PLAYER2, 0 for draw.
    """
    board = initialize_game_state()
    player = PLAYER1
    saved_state = {PLAYER1: None, PLAYER2: None}

    while True:
        if ((minimax_first and player == PLAYER1) or
            (not minimax_first and player == PLAYER2)):
            # Minimax's turn
            action, saved_state[player] = minimax.generate_move(
                board.copy(), player, saved_state[player], max_depth
            )
        else:
            # Random's turn
            action, saved_state[player] = random_agent.generate_move(
                board.copy(), player, saved_state[player]
            )
        apply_player_action(board, action, player)
        result = check_end_state(board, player)
        if result == GameState.IS_DRAW:
            return 0
        elif result == GameState.IS_WIN:
            return player
        elif result == GameState.IS_LOSS:
            return -player
        player = PLAYER1 if player == PLAYER2 else PLAYER2

def benchmark_winrate(num_games=100, max_depth=6):
    """
    Benchmark minimax agent win rate against random player.

    Plays multiple games alternating who goes first and calculates
    win rates, draw rates, and loss rates for the minimax agent.

    Parameters
    ----------
    num_games : int, optional
        Number of games to play. Default is 100.
    max_depth : int, optional
        Maximum search depth for minimax agent. Default is 6.

    Returns
    -------
    None
        Prints win rate statistics to console.
    """
    minimax_wins = 0
    draws = 0
    for i in range(num_games):
        minimax_first = (i % 2 == 0)
        winner = play_minimax_vs_random(minimax_first, max_depth)
        # Minimax is PLAYER1 if minimax_first else PLAYER2
        if ((winner == PLAYER1 and minimax_first) or
            (winner == PLAYER2 and not minimax_first)):
            minimax_wins += 1
        elif winner == 0:
            draws += 1
        # Print progress
        if (i+1) % 10 == 0:
            print(f"{i+1} games played...", flush=True)
    print(f"\nMinimax win rate: {minimax_wins/num_games:.2%}")
    print(f"Draw rate: {draws/num_games:.2%}")
    print(f"Random win rate: {(num_games - minimax_wins - draws)/num_games:.2%}")

if __name__ == "__main__":
    benchmark_winrate(num_games=100, max_depth=6)
