"""
Focused test to demonstrate symmetry benefits with forced symmetric scenarios.
"""

import numpy as np
import time
from agents.agent_minimax.minimax import MinimaxSavedState, generate_move_time_limited
from game_utils import BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2, BoardPiece


def create_forced_symmetric_scenario():
    """Create a scenario where we force the same position to be analyzed multiple times."""
    print("FOCUSED SYMMETRY BENEFIT TEST")
    print("=" * 50)
    
    # Create a simple asymmetric position
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=BoardPiece)
    board[5, 1] = PLAYER1  # Left side
    board[5, 2] = PLAYER2
    board[4, 1] = PLAYER2
    
    # Create the exact mirror
    mirrored = np.fliplr(board)
    
    print("Original board:")
    print(board)
    print("\nMirrored board:")
    print(mirrored)
    
    # Test 1: Use shared state (symmetry enabled)
    print("\n1. Testing with SHARED state (symmetry working)...")
    shared_state = MinimaxSavedState()
    
    start_time = time.time()
    # Analyze original position
    move1, shared_state = generate_move_time_limited(board, PLAYER1, shared_state, time_limit_secs=1.0)
    mid_time = time.time()
    
    # Analyze mirrored position with same state
    move2, shared_state = generate_move_time_limited(mirrored, PLAYER1, shared_state, time_limit_secs=1.0)
    end_time = time.time()
    
    time_shared = end_time - start_time
    time_first = mid_time - start_time
    time_second = end_time - mid_time
    
    print(f"  First analysis time: {time_first:.3f}s")
    print(f"  Second analysis time: {time_second:.3f}s")
    print(f"  Total time: {time_shared:.3f}s")
    print(f"  Moves: {move1} -> {move2}")
    
    tt_size_shared = len(shared_state.transposition_table) if isinstance(shared_state, MinimaxSavedState) else 0
    print(f"  Transposition table size: {tt_size_shared}")
    
    # Test 2: Use separate states (no symmetry benefit)
    print("\n2. Testing with SEPARATE states (no symmetry benefit)...")
    
    start_time = time.time()
    # Analyze original position
    move3, state1 = generate_move_time_limited(board, PLAYER1, None, time_limit_secs=1.0)
    mid_time = time.time()
    
    # Analyze mirrored position with fresh state
    move4, state2 = generate_move_time_limited(mirrored, PLAYER1, None, time_limit_secs=1.0)
    end_time = time.time()
    
    time_separate = end_time - start_time
    time_first_sep = mid_time - start_time
    time_second_sep = end_time - mid_time
    
    print(f"  First analysis time: {time_first_sep:.3f}s")
    print(f"  Second analysis time: {time_second_sep:.3f}s")
    print(f"  Total time: {time_separate:.3f}s")
    print(f"  Moves: {move3} -> {move4}")
    
    tt_size_separate = 0
    if isinstance(state1, MinimaxSavedState):
        tt_size_separate += len(state1.transposition_table)
    if isinstance(state2, MinimaxSavedState):
        tt_size_separate += len(state2.transposition_table)
    
    print(f"  Total transposition table entries: {tt_size_separate}")
    
    # Analysis
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Shared state approach: {time_shared:.3f}s")
    print(f"Separate state approach: {time_separate:.3f}s")
    print(f"Time difference: {time_separate - time_shared:.3f}s")
    print(f"Speedup: {time_separate / time_shared:.2f}x")
    
    print(f"\nMemory usage:")
    print(f"Shared state: {tt_size_shared} entries")
    print(f"Separate states: {tt_size_separate} entries")
    print(f"Memory efficiency: {tt_size_separate / max(tt_size_shared, 1):.2f}x")
    
    # Check if second analysis was faster with shared state
    if time_second < time_second_sep:
        print(f"\n✓ Second analysis was faster with shared state!")
        print(f"  Shared: {time_second:.3f}s vs Separate: {time_second_sep:.3f}s")
        print(f"  Improvement: {time_second_sep - time_second:.3f}s ({((time_second_sep - time_second) / time_second_sep * 100):.1f}%)")
    else:
        print(f"\n? Second analysis wasn't significantly faster with shared state")
    
    # Verify moves are correctly mirrored
    print(f"\nMove verification:")
    print(f"Original board moves: {move1} (shared) vs {move3} (separate)")
    print(f"Mirrored board moves: {move2} (shared) vs {move4} (separate)")
    
    # Check if moves are related by symmetry
    if move1 == move3 and move2 == move4:
        print("✓ Moves are consistent between approaches")
    else:
        print("? Moves differ between approaches")
    
    # Check if mirrored moves are actually mirrored
    expected_mirror = BOARD_COLS - 1 - int(move1)
    if int(move2) == expected_mirror:
        print(f"✓ Moves are properly mirrored: {move1} <-> {move2}")
    else:
        print(f"? Moves are not mirrored as expected: {move1} -> {move2} (expected {expected_mirror})")
    
    print(f"\n{'='*50}")
    print("CONCLUSION:")
    if time_shared < time_separate and time_second < time_second_sep:
        print("🎉 Symmetry improvements are providing clear benefits!")
    elif tt_size_shared < tt_size_separate:
        print("✓ Symmetry improvements are providing memory benefits!")
    else:
        print("ℹ️  Symmetry improvements are working correctly but benefits")
        print("   depend on the specific positions and search depth.")


def test_hash_collision_detection():
    """Test that different positions don't accidentally get the same hash."""
    print("\nTesting hash collision detection...")
    
    state = MinimaxSavedState()
    
    # Create several different positions
    positions = []
    
    # Empty board
    positions.append(np.zeros((BOARD_ROWS, BOARD_COLS), dtype=BoardPiece))
    
    # Different single pieces
    for col in range(BOARD_COLS):
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=BoardPiece)
        board[5, col] = PLAYER1
        positions.append(board)
    
    # Different two-piece combinations
    for col1 in range(BOARD_COLS):
        for col2 in range(col1 + 1, BOARD_COLS):
            board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=BoardPiece)
            board[5, col1] = PLAYER1
            board[5, col2] = PLAYER2
            positions.append(board)
    
    # Calculate hashes
    hashes = []
    for board in positions:
        hash_val = state.get_board_hash(board)
        hashes.append(hash_val)
    
    # Check for collisions
    unique_hashes = set(hashes)
    print(f"Generated {len(positions)} positions")
    print(f"Generated {len(hashes)} hashes")
    print(f"Unique hashes: {len(unique_hashes)}")
    
    if len(unique_hashes) == len(hashes):
        print("✓ No hash collisions detected")
    else:
        print(f"⚠️  {len(hashes) - len(unique_hashes)} hash collisions detected")
    
    # Test symmetry specifically
    symmetric_pairs = 0
    for board in positions:
        mirrored = np.fliplr(board)
        if not np.array_equal(board, mirrored):  # Only test actually different boards
            hash1 = state.get_board_hash(board)
            hash2 = state.get_board_hash(mirrored)
            if hash1 == hash2:
                symmetric_pairs += 1
    
    print(f"Symmetric pairs with same hash: {symmetric_pairs}")
    if symmetric_pairs > 0:
        print("✓ Symmetry is working - symmetric positions have same hash")


if __name__ == "__main__":
    create_forced_symmetric_scenario()
    test_hash_collision_detection()
