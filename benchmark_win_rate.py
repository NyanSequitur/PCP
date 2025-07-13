"""
Benchmark Minimax Agent vs Random Agent with Varying Depth and Time Limits

Output a CSV file `benchmark_winrate_vs_random.csv`
"""

import time
import csv
from typing import Tuple
from game_utils import (
    PLAYER1, PLAYER2, GameState,
    initialize_game_state, apply_player_action,
    check_end_state, check_move_status
)
from agents.agent_random.random import generate_move_random
from agents.agent_minimax.minimax import generate_move_time_limited
from agents.agent_minimax.saved_state import MinimaxSavedState


def play_minimax_vs_random(minimax_first: bool, max_depth: int, time_limit: float) -> Tuple[int, float]:
    """
    Play a game between a time-limited minimax agent and a random agent.

    Parameters
    ----------
    minimax_first : bool
        Whether the minimax agent plays first.
    max_depth : int
        Maximum depth allowed for minimax search.
    time_limit : float
        Time limit in seconds for the minimax agent per move.

    Returns
    -------
    Tuple[int, float]
        Outcome from minimax's perspective: 1 for win, 0 for draw, -1 for loss.
        Total time for the game.
    """
    board = initialize_game_state()
    saved_state = {PLAYER1: None, PLAYER2: None}
    player = PLAYER1
    start = time.time()

    while True:
        if ((minimax_first and player == PLAYER1) or
            (not minimax_first and player == PLAYER2)):
            move, saved_state[player] = generate_move_time_limited(
                board.copy(), player, saved_state[player],
                time_limit_secs=time_limit, max_depth=max_depth
            )
        else:
            move, saved_state[player] = generate_move_random(
                board.copy(), player, saved_state[player]
            )

        if check_move_status(board, move).name != "IS_VALID":
            return (-1 if ((minimax_first and player == PLAYER1) or
                           (not minimax_first and player == PLAYER2)) else 1), time.time() - start

        apply_player_action(board, move, player)
        end_state = check_end_state(board, player)

        if end_state == GameState.IS_WIN:
            return (1 if ((minimax_first and player == PLAYER1) or
                          (not minimax_first and player == PLAYER2)) else -1), time.time() - start
        elif end_state == GameState.IS_DRAW:
            return 0, time.time() - start

        player = PLAYER1 if player == PLAYER2 else PLAYER2


def benchmark_all(depths, times, num_games_per_setting=10, output_file="benchmark_winrate_vs_random.csv"):
    """Run win-rate and time-per-game benchmarks for minimax agent vs random agent.

    Parameters
    ----------
    depths : list of int
        List of max depths to benchmark.
    times : list of float
        List of time limits (in seconds) to benchmark.
    num_games_per_setting : int
        Number of games to run per configuration (default 10).
    output_file : str
        Filename for CSV output.
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["max_depth", "time_limit", "minimax_win_rate", "avg_game_time"])

        for max_depth in depths:
            for time_limit in times:
                print(f"\nTesting depth={max_depth}, time_limit={time_limit}s...")
                minimax_wins, draws, random_wins = 0, 0, 0
                total_time = 0.0

                for i in range(num_games_per_setting):
                    minimax_first = (i % 2 == 0)
                    result, game_time = play_minimax_vs_random(minimax_first, max_depth, time_limit)
                    total_time += game_time

                    if result == 1:
                        minimax_wins += 1
                        outcome = "Minimax win"
                    elif result == -1:
                        random_wins += 1
                        outcome = "Random win"
                    else:
                        draws += 1
                        outcome = "Draw"

                    print(f"  Game {i+1}/{num_games_per_setting}: {outcome} in {game_time:.2f}s")

                win_rate = minimax_wins / num_games_per_setting
                avg_time = total_time / num_games_per_setting

                writer.writerow([max_depth, time_limit, f"{win_rate:.2f}", f"{avg_time:.2f}"])
                print(f"  ✅ Win rate: {win_rate:.2%}, Avg time: {avg_time:.2f}s")


if __name__ == "__main__":
    benchmark_all(
        depths=[2, 4, 6, 8, 10],
        times=[1, 3, 5, 10],
        num_games_per_setting=10
    )
