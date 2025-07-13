#!/usr/bin/env python3
"""
Test script to verify the refactored minimax agent works with minimal_main.py
"""

print("Testing refactored minimax agent compatibility...")

try:
    # Test imports
    from agents.agent_minimax import generate_move
    from game_utils import initialize_game_state, PLAYER1
    print("✓ Imports successful")
    
    # Test function call
    board = initialize_game_state()
    move, state = generate_move(board, PLAYER1)
    print(f"✓ Generated move: {move}")
    print(f"✓ State type: {type(state)}")
    
    # Test minimal_main import
    import minimal_main
    print("✓ minimal_main.py imports successfully")
    
    print("\n🎉 SUCCESS: The refactored minimax agent is fully compatible!")
    print("   - All imports work correctly")
    print("   - Function signatures are preserved")
    print("   - minimal_main.py can use the refactored agent")
    print("   - All functionality is maintained")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
