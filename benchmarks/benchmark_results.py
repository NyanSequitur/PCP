"""
Data structures for benchmark results and statistics.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any

from agents.agent_minimax.transposition_table import TranspositionTable
from agents.agent_minimax.saved_state import MinimaxSavedState


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    description: str
    execution_time: float
    iterations: int
    results: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric(self, key: str, value: Any):
        """Add a metric to the results."""
        self.results[key] = value
    
    def get_rate(self) -> float:
        """Get operations per second."""
        return self.iterations / self.execution_time if self.execution_time > 0 else 0


@dataclass 
class TranspositionTableStats:
    """Statistics for transposition table benchmarks."""
    hit_rate: float
    lookups: int
    hits: int
    stores: int
    collisions: int
    table_size: int
    max_table_size: int
    
    @classmethod
    def from_tt(cls, tt: TranspositionTable) -> 'TranspositionTableStats':
        """Create stats from a transposition table."""
        return cls(
            hit_rate=tt.get_hit_rate(),
            lookups=tt.lookups,
            hits=tt.hits,
            stores=tt.stores,
            collisions=tt.collisions,
            table_size=len(tt),
            max_table_size=tt.max_table_size
        )


class StatsTracker:
    """Simple statistics tracker for benchmarking."""
    def __init__(self):
        self.lookups = 0
        self.hits = 0
        self.stores = 0


class StatTrackingWrapper:
    """Wrapper for MinimaxSavedState that tracks statistics."""
    def __init__(self, state: MinimaxSavedState):
        self.state = state
        self.stats = StatsTracker()
    
    def store_position(self, board, depth: int, value: float,
                      best_move, flag: str):
        self.stats.stores += 1
        return self.state.store_position(board, depth, value, best_move, flag)
    
    def lookup_position(self, board, depth: int, alpha: float, beta: float):
        self.stats.lookups += 1
        found, value = self.state.lookup_position(board, depth, alpha, beta)
        if found:
            self.stats.hits += 1
        return found, value
    
    def __getattr__(self, name):
        """Delegate other attributes to the wrapped state."""
        return getattr(self.state, name)


class TranspositionTableNoSymmetry(TranspositionTable):
    """TranspositionTable that doesn't use symmetry optimization."""
    
    def _get_canonical_board(self, board):
        """Override to disable symmetry - always return original board."""
        return board
    
    def _is_board_mirrored(self, board) -> bool:
        """Override to never consider board mirrored."""
        return False


def expected_nodes_for_depth(depth: int) -> int:
    """Estimate expected nodes for a given search depth."""
    # Rough estimate based on branching factor of ~7 for Connect Four
    return 7 ** depth
