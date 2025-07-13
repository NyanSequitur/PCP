#!/usr/bin/env python3
"""
Enhanced proof focusing on why symmetry optimization matters in practice.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, 
    PLAYER1, PLAYER2, check_end_state, GameState
)
from agents.agent_minimax.transposition_table import TranspositionTable
from agents.agent_minimax.search import negamax_search
from agents.agent_minimax.move_ordering import get_valid_columns
from benchmarks.benchmark_results import TranspositionTableNoSymmetry


def enhanced_real_search_proof():
    """Enhanced proof showing why the real search matters."""
    print("🔬 ENHANCED REAL SEARCH PROOF")
    print("=" * 60)
    
    # Start with an empty board (maximum symmetry potential)
    board = initialize_game_state()
    print("Starting from empty board (maximum symmetry potential):")
    print(board)
    print()
    
    # Test different search depths
    depths = [3, 4, 5, 8, 10]
    
    for depth in depths:
        print(f"🔍 Testing depth {depth} search...")
        
        # Search with symmetry
        tt_with = TranspositionTable()
        start_time = time.perf_counter()
        result_with = negamax_search(board, depth, -1000, 1000, PLAYER1, tt_with, 
                                   time.monotonic() + 30.0)
        time_with = time.perf_counter() - start_time
        
        # Search without symmetry
        tt_without = TranspositionTable()
        # Force it to not use symmetry by replacing with no-symmetry version
        original_get_canonical = tt_without._get_canonical_board
        tt_without._get_canonical_board = lambda board: board  # Disable symmetry
        
        start_time = time.perf_counter()
        result_without = negamax_search(board, depth, -1000, 1000, PLAYER1, tt_without,
                                      time.monotonic() + 30.0)
        time_without = time.perf_counter() - start_time
        
        print(f"  Depth {depth} results:")
        print(f"    Same result: {'✅' if abs(result_with - result_without) < 0.01 else '❌'}")
        print(f"    Table size WITH symmetry:    {len(tt_with)}")
        print(f"    Table size WITHOUT symmetry: {len(tt_without)}")
        
        if len(tt_without) > len(tt_with):
            reduction = (len(tt_without) - len(tt_with)) / len(tt_without) * 100
            print(f"    Memory reduction: {reduction:.1f}% ({len(tt_without) - len(tt_with)} entries saved)")
        else:
            print(f"    No reduction observed")
        
        print(f"    Time WITH symmetry:    {time_with:.3f}s")
        print(f"    Time WITHOUT symmetry: {time_without:.3f}s")
        print()


def position_analysis_proof():
    """Analyze what types of positions actually get stored."""
    print("🔬 POSITION ANALYSIS PROOF")
    print("=" * 60)
    
    # Create a position that will generate symmetric sub-positions
    board = initialize_game_state()
    
    tt_with = TranspositionTable()
    tt_without = TranspositionTableNoSymmetry()
    
    # Manually explore some positions that should be symmetric
    symmetric_test_positions = []
    
    # Generate first-level moves (these should be symmetric)
    for col in range(7):
        test_board = board.copy()
        apply_player_action(test_board, PlayerAction(col), PLAYER1)
        
        # Store this position
        tt_with.store_position(test_board, 3, float(col), PlayerAction(col), 'exact')
        tt_without.store_position(test_board, 3, float(col), PlayerAction(col), 'exact')
        
        # Check if this has a symmetric counterpart
        mirror_col = 6 - col
        if mirror_col != col:  # Skip center column
            mirror_board = board.copy()
            apply_player_action(mirror_board, PlayerAction(mirror_col), PLAYER1)
            
            # Store the mirror
            tt_with.store_position(mirror_board, 3, float(mirror_col), PlayerAction(mirror_col), 'exact')
            tt_without.store_position(mirror_board, 3, float(mirror_col), PlayerAction(mirror_col), 'exact')
            
            # Check if they map to same hash
            hash_with_orig = tt_with.get_board_hash(test_board)
            hash_with_mirror = tt_with.get_board_hash(mirror_board)
            hash_without_orig = tt_without.get_board_hash(test_board)
            hash_without_mirror = tt_without.get_board_hash(mirror_board)
            
            print(f"Column {col} ↔ Column {mirror_col}:")
            print(f"  With symmetry:    {'SAME' if hash_with_orig == hash_with_mirror else 'DIFFERENT'} hash")
            print(f"  Without symmetry: {'SAME' if hash_without_orig == hash_without_mirror else 'DIFFERENT'} hash")
    
    print(f"\nAfter storing symmetric first moves:")
    print(f"  Table size WITH symmetry:    {len(tt_with)}")
    print(f"  Table size WITHOUT symmetry: {len(tt_without)}")
    print(f"  Reduction factor: {len(tt_without) / len(tt_with):.2f}x")


def memory_usage_proof():
    """Demonstrate actual memory impact with large datasets."""
    print("\n🔬 MEMORY USAGE PROOF") 
    print("=" * 60)
    
    # Create large numbers of symmetric pairs to show real memory impact
    print("Creating large datasets to demonstrate memory impact...")
    
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        # Generate symmetric board pairs
        pairs = []
        for i in range(size):
            # Create asymmetric pattern
            board = initialize_game_state()
            moves = [i % 3, (i+1) % 3, (i+2) % 3]  # Create variation
            for j, col in enumerate(moves):
                if col < 7:  # Valid column
                    try:
                        apply_player_action(board, PlayerAction(col), PLAYER1 if j % 2 == 0 else PLAYER2)
                    except:
                        break
            
            mirror = np.fliplr(board)
            if not np.array_equal(board, mirror):  # Only add if asymmetric
                pairs.append((board, mirror))
        
        # Test memory usage
        tt_with = TranspositionTable()
        tt_without = TranspositionTableNoSymmetry()
        
        for board, mirror in pairs:
            tt_with.store_position(board, 5, 100.0, PlayerAction(3), 'exact')
            tt_with.store_position(mirror, 5, 101.0, PlayerAction(3), 'exact')
            
            tt_without.store_position(board, 5, 100.0, PlayerAction(3), 'exact') 
            tt_without.store_position(mirror, 5, 101.0, PlayerAction(3), 'exact')
        
        memory_saved = len(tt_without) - len(tt_with)
        reduction_pct = memory_saved / len(tt_without) * 100 if len(tt_without) > 0 else 0
        
        print(f"Dataset size {size}:")
        print(f"  Valid symmetric pairs: {len(pairs)}")
        print(f"  With symmetry:    {len(tt_with)} entries")
        print(f"  Without symmetry: {len(tt_without)} entries")
        print(f"  Memory saved:     {memory_saved} entries ({reduction_pct:.1f}%)")
        print()


if __name__ == "__main__":
    enhanced_real_search_proof()
    position_analysis_proof()
    memory_usage_proof()