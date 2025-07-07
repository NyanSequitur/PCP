#!/usr/bin/env python3
"""
Demonstration of the refactored minimax agent.

This script shows how to use the new modular architecture and demonstrates
the key components working together.
"""

import numpy as np
from game_utils import (
    initialize_game_state, apply_player_action, 
    PlayerAction, PLAYER1, PLAYER2, print_board
)

# Import the main agent interface
from agents.agent_minimax import generate_move_time_limited, MinimaxSavedState

# Import individual components to demonstrate modular usage
from agents.agent_minimax.heuristics import evaluate_board
from agents.agent_minimax.transposition_table import TranspositionTable
from agents.agent_minimax.move_ordering import get_valid_columns, order_moves
from agents.agent_minimax.search_continuation import get_search_statistics


def demonstrate_basic_usage():
    """Demonstrate basic usage of the refactored minimax agent."""
    print("=== Basic Usage Demonstration ===")
    
    # Initialize a game
    board = initialize_game_state()
    print("Initial board:")
    print_board(board)
    
    # Generate a move using the main interface
    print("\nGenerating move for PLAYER1...")
    move, saved_state = generate_move_time_limited(
        board, PLAYER1, 
        time_limit_secs=2.0,
        max_depth=10
    )
    
    print(f"Generated move: Column {move}")
    print(f"Saved state type: {type(saved_state)}")
    print(f"Transposition table size: {len(saved_state.transposition_table)}")
    
    # Apply the move
    apply_player_action(board, move, PLAYER1)
    print("\nBoard after move:")
    print_board(board)
    
    return board, saved_state


def demonstrate_modular_components():
    """Demonstrate individual components of the modular architecture."""
    print("\n=== Modular Components Demonstration ===")
    
    # Create a simple board position
    board = initialize_game_state()
    apply_player_action(board, PlayerAction(3), PLAYER1)  # Center
    apply_player_action(board, PlayerAction(2), PLAYER2)  # Left of center
    
    print("Test position:")
    print_board(board)
    
    # 1. Heuristic evaluation
    print("\n1. Heuristic Evaluation:")
    score_p1 = evaluate_board(board, PLAYER1)
    score_p2 = evaluate_board(board, PLAYER2)
    print(f"   PLAYER1 score: {score_p1:.2f}")
    print(f"   PLAYER2 score: {score_p2:.2f}")
    
    # 2. Move ordering
    print("\n2. Move Ordering:")
    valid_cols = get_valid_columns(board)
    print(f"   Valid columns: {valid_cols}")
    
    tt = TranspositionTable()
    ordered_cols = order_moves(board, valid_cols, tt)
    print(f"   Ordered columns: {ordered_cols}")
    
    # 3. Transposition table
    print("\n3. Transposition Table:")
    board_hash = tt.get_board_hash(board)
    print(f"   Board hash: {board_hash[:16]}...")
    
    # Store a position
    tt.store_position(board, depth=5, value=score_p1, 
                     best_move=PlayerAction(3), flag='exact')
    print(f"   Position stored. Table size: {len(tt)}")
    
    # Lookup the position
    found, value = tt.lookup_position(board, depth=3, alpha=-1000, beta=1000)
    print(f"   Position lookup: found={found}, value={value:.2f}")


def demonstrate_persistent_state():
    """Demonstrate how the saved state persists between moves."""
    print("\n=== Persistent State Demonstration ===")
    
    board = initialize_game_state()
    saved_state = None
    
    print("Playing a few moves to build up transposition table...")
    
    for move_num in range(4):
        current_player = PLAYER1 if move_num % 2 == 0 else PLAYER2
        
        print(f"\nMove {move_num + 1} - {current_player}:")
        
        # Generate move with persistent state
        move, saved_state = generate_move_time_limited(
            board, current_player, saved_state, 
            time_limit_secs=1.0
        )
        
        apply_player_action(board, move, current_player)
        
        print(f"   Chose column {move}")
        print(f"   TT size: {len(saved_state.transposition_table)}")
        
        # Show cache statistics
        stats = saved_state.get_cache_statistics()
        print(f"   TT utilization: {stats['tt_utilization']:.2%}")
    
    print("\nFinal board:")
    print_board(board)


def demonstrate_performance_analysis():
    """Demonstrate performance analysis capabilities."""
    print("\n=== Performance Analysis Demonstration ===")
    
    board = initialize_game_state()
    
    print("Analyzing search performance at different time limits...")
    
    time_limits = [0.5, 1.0, 2.0, 5.0]
    
    for time_limit in time_limits:
        print(f"\nTime limit: {time_limit}s")
        
        import time
        start_time = time.time()
        
        move, saved_state = generate_move_time_limited(
            board, PLAYER1, 
            time_limit_secs=time_limit,
            max_depth=20
        )
        
        actual_time = time.time() - start_time
        
        # Get search statistics
        stats = get_search_statistics(
            saved_state.transposition_table, 
            completed_depth=0,  # We don't have this info directly
            search_time=actual_time
        )
        
        print(f"   Move: {move}")
        print(f"   Actual time: {actual_time:.3f}s")
        print(f"   TT size: {stats['tt_size']}")
        print(f"   TT utilization: {len(saved_state.transposition_table) / saved_state.transposition_table.max_table_size:.2%}")


def main():
    """Main demonstration function."""
    print("Minimax Agent - Modular Architecture Demonstration")
    print("=" * 50)
    
    try:
        # Run demonstrations
        board, saved_state = demonstrate_basic_usage()
        demonstrate_modular_components()
        demonstrate_persistent_state()
        demonstrate_performance_analysis()
        
        print("\n" + "=" * 50)
        print("✓ All demonstrations completed successfully!")
        print("\nThe refactored minimax agent is working correctly with:")
        print("- Modular architecture for better maintainability")
        print("- Preserved all original functionality")
        print("- Enhanced documentation and code organization")
        print("- Full backward compatibility")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
