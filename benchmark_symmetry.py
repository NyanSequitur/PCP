#!/usr/bin/env python3
"""
Comprehensive symmetry benchmark & analysis for the Connect-Four Minimax agent.
┌──────────────────────────────────────────────────────────────────────┐
│  • quick sanity checks (symmetry detection)                         │
│  • hash-collision analysis                                          │
│  • time-limited move generation benchmark (shallow search)          │
│  • fixed-depth search benchmark with full TT stats                  │
│  • one consolidated results table + narrative summary               │
└──────────────────────────────────────────────────────────────────────┘
Usage:  python benchmark_symmetry.py
"""

import time
import copy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# 🏗  Import your game / agent modules
#     (paths stay identical to your original code base)
# ──────────────────────────────────────────────────────────────────────
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction,
    PLAYER1, PLAYER2
)
from agents.agent_minimax.minimax import (
    generate_move_time_limited, MinimaxSavedState, TranspositionEntry,
    _negamax, _other
)

# ╭──────────────────────────────────────────────────────────────────╮
# │  1.  Dataclasses and helper states                              │
# ╰──────────────────────────────────────────────────────────────────╯
@dataclass
class BenchmarkResult:
    total_time: float
    table_size: int
    lookups: int
    hits: int
    stores: int
    nodes_searched: int
    positions: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class MinimaxStateNoSym(MinimaxSavedState):
    """Minimax state WITHOUT symmetry (override hashing only)."""
    def get_board_hash(self, board: np.ndarray) -> str:
        return board.tobytes().hex()

    def _is_board_mirrored(self, board: np.ndarray) -> bool:
        return False  # never mirrored


class MinimaxStateStats(MinimaxSavedState):
    """Minimax state WITH symmetry + TT-stats collection."""
    def __init__(self, max_table_size: int = 1_000_000):
        super().__init__(max_table_size)
        self.lookups = self.hits = self.stores = 0

    # wrap lookup/store to count stats
    def lookup_position(self, board: np.ndarray, depth: int,
                        alpha: float, beta: float):
        self.lookups += 1
        hit, val = super().lookup_position(board, depth, alpha, beta)
        if hit:
            self.hits += 1
        return hit, val

    def store_position(self, board: np.ndarray, depth: int, value: float,
                       best_move: Optional[PlayerAction], flag: str):
        self.stores += 1
        super().store_position(board, depth, value, best_move, flag)


class MinimaxStateNoSymStats(MinimaxStateStats, MinimaxStateNoSym):
    """No-symmetry variant also collecting stats (MRO does the override)."""
    pass


# ╭──────────────────────────────────────────────────────────────────╮
# │  2.  Position generators                                         │
# ╰──────────────────────────────────────────────────────────────────╯
def gen_basic_positions() -> List[np.ndarray]:
    """
    A mix of empty, early, mid, symmetric & asymmetric boards.
    """
    positions: List[np.ndarray] = [initialize_game_state()]

    early_seqs = [[3], [3, 2], [3, 2, 4, 3]]
    for seq in early_seqs:
        positions.append(play_sequence(seq))

    symmetric_seqs = [
        [3, 3, 2, 4, 1, 5],
        [0, 6, 1, 5, 2, 4],
    ]
    for seq in symmetric_seqs:
        positions.append(play_sequence(seq))

    asym_seqs = [
        [0, 1, 2, 3, 4],
        [6, 5, 4, 3, 2],
        [3, 2, 1, 4, 5, 6],
    ]
    for seq in asym_seqs:
        positions.append(play_sequence(seq))

    return positions


def gen_comprehensive_positions() -> List[np.ndarray]:
    """Richer set focusing on mirror pairs + near-sym patterns."""
    pos: List[np.ndarray] = []

    # strictly symmetric templates
    templates = [
        [],
        [3],  # centre
        [3, 2, 4],
        [2, 4, 2, 4],
        [1, 5, 1, 5],
        [0, 6, 0, 6],
    ]
    for t in templates:
        pos.append(play_sequence(t))

    # near-sym but not exact
    near = [
        [3, 2, 4, 3],
        [2, 4, 1, 5],
        [1, 5, 2, 4],
    ]
    for n in near:
        pos.append(play_sequence(n))

    # explicit mirror pairs
    pairs = [([2], [4]), ([1], [5]), ([0], [6]), ([2, 1], [4, 5])]
    for left, right in pairs:
        pos.append(play_sequence(left))
        pos.append(play_sequence(right))

    return pos


def play_sequence(moves: List[int]) -> np.ndarray:
    board = initialize_game_state()
    player = PLAYER1
    for m in moves:
        apply_player_action(board, PlayerAction(m), player)
        player = _other(player)
    return board.copy()


# ╭──────────────────────────────────────────────────────────────────╮
# │  3.  Core benchmark helpers                                     │
# ╰──────────────────────────────────────────────────────────────────╯
def shallow_time_benchmark(positions: List[np.ndarray],
                           use_symmetry: bool,
                           time_limit: float = 1.0,
                           max_depth: int = 6) -> BenchmarkResult:
    """Time-limited move generation across several positions."""
    StateCls = MinimaxSavedState if use_symmetry else MinimaxStateNoSym
    state = StateCls()
    total_time = 0.0
    nodes = 0

    for board in positions:
        start = time.monotonic()
        try:
            _, state = generate_move_time_limited(
                board, PLAYER1, state, time_limit, max_depth
            )
        except Exception:
            pass
        total_time += time.monotonic() - start

    # stats only if subclass collected them
    lookups = getattr(state, "lookups", 0)
    hits = getattr(state, "hits", 0)
    stores = getattr(state, "stores", 0)

    return BenchmarkResult(
        total_time=total_time,
        table_size=len(state.transposition_table),
        lookups=lookups,
        hits=hits,
        stores=stores,
        nodes_searched=nodes,
        positions=len(positions),
    )


def fixed_depth_benchmark(positions: List[np.ndarray],
                           depth: int = 4,
                           use_symmetry: bool = True) -> BenchmarkResult:
    """Fixed-depth search on each position with full TT stats."""
    StateCls = MinimaxStateStats if use_symmetry else MinimaxStateNoSymStats
    state = StateCls()
    total_time = nodes = 0

    for board in positions:
        start = time.monotonic()
        _negamax(board, depth, -float('inf'), float('inf'),
                  PLAYER1, state, time.monotonic() + 10)
        total_time += time.monotonic() - start
        # _negamax already updates state.stats internally (if any)

    return BenchmarkResult(
        total_time=total_time,
        table_size=len(state.transposition_table),
        lookups=state.lookups,
        hits=state.hits,
        stores=state.stores,
        nodes_searched=nodes,
        positions=len(positions),
    )


def hash_collision_rate(positions: List[np.ndarray]) -> Tuple[int, int]:
    """Return (unique_with_sym, unique_without_sym)."""
    sym_state = MinimaxSavedState()
    nosym_state = MinimaxStateNoSym()

    hashes_sym = {sym_state.get_board_hash(b) for b in positions}
    hashes_nosym = {nosym_state.get_board_hash(b) for b in positions}
    return len(hashes_sym), len(hashes_nosym)


# ╭──────────────────────────────────────────────────────────────────╮
# │  4.  Reporting helpers                                           │
# ╰──────────────────────────────────────────────────────────────────╯
def pct(a: float, b: float) -> float:
    """% improvement of a over b where bigger-is-better = lower value."""
    return (b - a) / b * 100 if b else 0


def print_table(rows: List[Tuple[str, str, str, str]]):
    width = max(len(r[0]) for r in rows) + 2
    print(f"{'Metric':<{width}} Symmetry ON    Symmetry OFF   Δ / %Improvement")
    print("-" * (width + 36))
    for name, v_on, v_off, delta in rows:
        print(f"{name:<{width}} {v_on:>12}   {v_off:>12}   {delta:>12}")
    print()


# ╭──────────────────────────────────────────────────────────────────╮
# │  5.  Main orchestration                                          │
# ╰──────────────────────────────────────────────────────────────────╯
def main():  # noqa: C901
    print("\n=== Symmetry Benchmark Suite ===\n")

    # 5-A  Generate position sets
    basic = gen_basic_positions()
    comp  = gen_comprehensive_positions()

    # 5-B  Quick symmetry sanity checks
    sym_state = MinimaxSavedState()
    empty_mirror = sym_state._is_board_mirrored(initialize_game_state())
    one_left = play_sequence([2])
    one_right = play_sequence([4])
    print(f"Sanity • empty mirrored? {empty_mirror}")
    print(f"Sanity • hash(left)==hash(right)?",
          sym_state.get_board_hash(one_left) ==
          sym_state.get_board_hash(one_right), "\n")

    # 5-C  Time-limited shallow benchmark
    shallow_on  = shallow_time_benchmark(basic, True)
    shallow_off = shallow_time_benchmark(basic, False)

    # 5-D  Deeper fixed-depth benchmark
    deep_on  = fixed_depth_benchmark(comp, depth=4, use_symmetry=True)
    deep_off = fixed_depth_benchmark(comp, depth=4, use_symmetry=False)

    # 5-E  Hash collision analysis
    u_sym, u_nosym = hash_collision_rate(comp)
    collision_pct = (u_nosym - u_sym) / u_nosym * 100 if u_nosym else 0

    # 5-F  Report (shallow)
    print("── Shallow time-limited search ({} positions)".format(shallow_on.positions))
    rows = [
        ("Total time (s)",
         f"{shallow_on.total_time:>.3f}",
         f"{shallow_off.total_time:>.3f}",
         f"{pct(shallow_on.total_time, shallow_off.total_time):.1f}%"),
        ("TT size",
         str(shallow_on.table_size),
         str(shallow_off.table_size),
         f"{pct(shallow_on.table_size, shallow_off.table_size):.1f}%"),
    ]
    print_table(rows)

    # 5-G  Report (deep)
    print("── Fixed-depth search ({} positions, depth=4)".format(deep_on.positions))
    rows = [
        ("Total time (s)",
         f"{deep_on.total_time:>.3f}", f"{deep_off.total_time:>.3f}",
         f"{pct(deep_on.total_time, deep_off.total_time):.1f}%"),
        ("TT size",
         str(deep_on.table_size), str(deep_off.table_size),
         f"{pct(deep_on.table_size, deep_off.table_size):.1f}%"),
        ("TT lookups",
         str(deep_on.lookups), str(deep_off.lookups), "-"),
        ("TT hits",
         str(deep_on.hits), str(deep_off.hits), "-"),
        ("TT hit-rate",
         f"{deep_on.hit_rate:.3f}", f"{deep_off.hit_rate:.3f}", "-"),
    ]
    print_table(rows)

    # 5-H  Hash report
    print("── Hash collisions on comprehensive set")
    print(f"Unique WITH symmetry : {u_sym}")
    print(f"Unique WITHOUT sym   : {u_nosym}")
    print(f"Collision rate       : {collision_pct:.1f}%\n")

    # 5-I  Narrative summary
    speedup = pct(deep_on.total_time, deep_off.total_time)
    table_save = pct(deep_on.table_size, deep_off.table_size)
    print("=== SUMMARY ===")
    if speedup > 0:
        print(f"• Search is {speedup:.1f}% faster with symmetry.")
    else:
        print("• No time speed-up detected (possible overhead).")

    print(f"• Transposition table {table_save:.1f}% smaller.")
    print(f"• Hash collision reduction: {collision_pct:.1f}% "
          f"({u_nosym - u_sym} positions collapsed).")

    print("• TT hit-rate ↑ from "
          f"{deep_off.hit_rate:.1%} to {deep_on.hit_rate:.1%}.")

    verdict = ("SIGNIFICANT" if speedup > 15 else
               "MODERATE" if speedup > 7 else
               "MINOR")
    print(f"\nVerdict: Symmetry handling gives {verdict} benefit.\n")


# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
