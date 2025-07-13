"""
Benchmarking Minimax Agent Performance: Time vs. TT Size, varying:

    - max_depth values
    - time_limit values (seconds)
    - board fill levels (0%, 25%, 50%, 75%)

The results are saved to a CSV file with the following columns:
    - fill_pct
    - max_depth
    - time_limit
    - actual_time_used:
    - tt_size
"""

import time
import csv
import random
from typing import List
import numpy as np

from game_utils import (
    initialize_game_state,
    apply_player_action,
    PlayerAction,
    PLAYER1,
)
from agents.agent_minimax.minimax import generate_move_time_limited
from agents.agent_minimax.saved_state import MinimaxSavedState


def generate_random_board(fill_pct: float) -> np.ndarray:
    """
    Generate a board randomly filled to a given percentage.

    Parameters
    ----------
    fill_pct : float
        The fraction of total board cells to fill with alternating player moves.

    Returns
    -------
    np.ndarray
        A valid game board filled to certain percentage
    """
    board = initialize_game_state()
    total_cells = board.shape[0] * board.shape[1]
    num_moves = int(fill_pct * total_cells)

    player = PLAYER1
    move_count = 0
    while move_count < num_moves:
        valid_cols = [c for c in range(board.shape[1]) if board[0, c] == 0]
        if not valid_cols:
            break
        col = random.choice(valid_cols)
        try:
            apply_player_action(board, PlayerAction(col), player)
            move_count += 1
            player = PLAYER1 if player == -PLAYER1 else -PLAYER1
        except Exception:
            continue
    return board


def benchmark_time_and_tt(
    max_depth_list: List[int],
    time_limits: List[int],
    fill_levels: List[float],
    output_file: str = "benchmark_minimax_time_tt.csv"
):
    """
    Benchmark minimax performance across different parameters.

    Tests the minimax agent with various configurations of max depth,
    time limits, and board fill levels, recording performance metrics.

    Parameters
    ----------
    max_depth_list : List[int]
        List of maximum search depths to test.
    time_limits : List[int]
        List of time limits (in seconds) to test.
    fill_levels : List[float]
        List of board fill percentages (0.0 to 1.0) to test.
    output_file : str, optional
        CSV file to write results to. Default is "benchmark_minimax_time_tt.csv".

    Returns
    -------
    None
        Results are written to the specified CSV file.
    """
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "fill_pct", "max_depth", "time_limit", "actual_time_used", "tt_size"
        ])

        for fill_pct in fill_levels:
            for depth in max_depth_list:
                for tmax in time_limits:
                    board = generate_random_board(fill_pct)
                    state = MinimaxSavedState()
                    start = time.time()
                    try:
                        _, state = generate_move_time_limited(
                            board,
                            PLAYER1,
                            state,
                            time_limit_secs=tmax,
                            max_depth=depth
                        )
                    except Exception:
                        continue
                    end = time.time()
                    elapsed = end - start
                    tt_size = len(state.transposition_table)
                    writer.writerow([fill_pct, depth, tmax, elapsed, tt_size])
                    print(f"[✓] fill={fill_pct:.0%}, depth={depth}, tmax={tmax}s → {elapsed:.2f}s, TT={tt_size}")


if __name__ == "__main__":
    benchmark_time_and_tt(
        max_depth_list=[6, 8, 10, 12],
        time_limits=[1, 5, 10, 15],
        fill_levels=[0.0, 0.25, 0.5, 0.75]
    )
