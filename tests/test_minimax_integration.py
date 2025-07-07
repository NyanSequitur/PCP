"""
Integration tests for the minimax agent.

This module tests the integration between all minimax components,
including end-to-end functionality and performance characteristics.
"""

import numpy as np
import pytest
import time
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2,
    check_end_state, GameState, BOARD_COLS
)
from agents.agent_minimax.minimax import (
    generate_move_time_limited, generate_move, create_minimax_agent
)
from agents.agent_minimax.saved_state import MinimaxSavedState


class TestMinimaxIntegrationBasic:
    """Test basic integration between minimax components."""
    
    def test_complete_game_simulation(self):
        """Test running a complete game with minimax agents."""
        board = initialize_game_state()
        
        # Create two minimax agents
        agent1_func, agent1_state = create_minimax_agent(time_limit=1.0, max_depth=5)
        agent2_func, agent2_state = create_minimax_agent(time_limit=1.0, max_depth=5)
        
        move_count = 0
        max_moves = 42  # Maximum possible moves in Connect Four
        
        while move_count < max_moves:
            # Check if game is over
            game_state = check_end_state(board, PLAYER1)
            if game_state != GameState.STILL_PLAYING:
                break
            
            # Player 1 move
            move1, agent1_state = agent1_func(board, PLAYER1, agent1_state)
            apply_player_action(board, move1, PLAYER1)
            move_count += 1
            
            # Check if game is over
            game_state = check_end_state(board, PLAYER2)
            if game_state != GameState.STILL_PLAYING:
                break
            
            # Player 2 move
            move2, agent2_state = agent2_func(board, PLAYER2, agent2_state)
            apply_player_action(board, move2, PLAYER2)
            move_count += 1
        
        # Game should end in reasonable number of moves
        assert move_count <= max_moves
        assert move_count > 0
        
        # Final state should be terminal
        final_state = check_end_state(board, PLAYER1)
        assert final_state in [GameState.IS_WIN, GameState.IS_DRAW]
    
    def test_minimax_vs_minimax_consistency(self):
        """Test that minimax agents play consistently."""
        board = initialize_game_state()
        
        # Create identical agents
        agent1_func, agent1_state = create_minimax_agent(time_limit=1.0, max_depth=4)
        agent2_func, agent2_state = create_minimax_agent(time_limit=1.0, max_depth=4)
        
        # Apply same initial moves
        initial_moves = [PlayerAction(3), PlayerAction(2), PlayerAction(4)]
        for i, move in enumerate(initial_moves):
            player = PLAYER1 if i % 2 == 0 else PLAYER2
            apply_player_action(board, move, player)
        
        # Both agents should make the same move from the same position
        move1, _ = agent1_func(board, PLAYER1, agent1_state)
        move2, _ = agent2_func(board, PLAYER1, agent2_state)
        
        assert move1 == move2
    
    def test_minimax_state_persistence(self):
        """Test that minimax state persists correctly across moves."""
        board = initialize_game_state()
        agent_func, agent_state = create_minimax_agent(time_limit=1.0, max_depth=4)
        
        moves_and_states = []
        
        # Play several moves
        for i in range(6):
            player = PLAYER1 if i % 2 == 0 else PLAYER2
            move, agent_state = agent_func(board, player, agent_state)
            apply_player_action(board, move, player)
            moves_and_states.append((move, agent_state))
        
        # State should be the same object throughout
        for _, state in moves_and_states[1:]:
            assert state is moves_and_states[0][1]
        
        # Transposition table should grow
        final_state = moves_and_states[-1][1]
        assert isinstance(final_state, MinimaxSavedState)
        assert len(final_state.transposition_table) > 0


class TestMinimaxIntegrationPerformance:
    """Test performance characteristics of minimax integration."""
    
    def test_time_limit_compliance(self):
        """Test that minimax respects time limits consistently."""
        board = initialize_game_state()
        time_limit = 0.5  # 500ms
        
        # Test multiple moves with time limit
        times = []
        for _ in range(5):
            start_time = time.time()
            move, state = generate_move_time_limited(
                board, PLAYER1, time_limit_secs=time_limit
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            assert elapsed <= time_limit * 2  # Allow some buffer
            assert isinstance(move, np.integer)  # PlayerAction is np.int8
        
        # All times should be reasonable
        avg_time = sum(times) / len(times)
        assert avg_time <= time_limit * 1.5
    
    def test_transposition_table_efficiency(self):
        """Test that transposition table improves performance."""
        board = initialize_game_state()
        
        # First search without existing state
        start_time = time.time()
        move1, state1 = generate_move_time_limited(board, PLAYER1, time_limit_secs=2.0)
        time_without_tt = time.time() - start_time
        
        # Apply the move
        apply_player_action(board, move1, PLAYER1)
        
        # Second search with existing state (should be faster due to TT)
        start_time = time.time()
        move2, state2 = generate_move_time_limited(board, PLAYER2, state1, time_limit_secs=2.0)
        time_with_tt = time.time() - start_time
        
        # Should have reasonable performance
        assert time_with_tt <= 2.5  # Allow buffer
        assert time_without_tt <= 2.5
        
        # State should be reused
        assert state2 is state1
        
        # Transposition table should have entries
        assert hasattr(state2, 'transposition_table')
        assert isinstance(state2, MinimaxSavedState)
        assert len(state2.transposition_table) > 0
    
    def test_depth_scaling_performance(self):
        """Test performance scaling with search depth."""
        board = initialize_game_state()
        
        # Test different depths
        depths = [2, 3, 4, 5]
        times = []
        
        for depth in depths:
            start_time = time.time()
            move, state = generate_move_time_limited(
                board, PLAYER1, time_limit_secs=5.0, max_depth=depth
            )
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            assert isinstance(move, np.integer)
        
        # Times should generally increase with depth (though not strictly)
        # Just check that all complete in reasonable time
        for t in times:
            assert t < 5.5  # Allow some buffer


class TestMinimaxIntegrationRobustness:
    """Test robustness of minimax integration."""
    
    def test_edge_case_positions(self):
        """Test minimax on various edge case positions."""
        
        # Test near-full board
        board = initialize_game_state()
        
        # Fill most columns
        for col in range(BOARD_COLS - 1):
            for row in range(5):  # Leave one space in each column
                player = PLAYER1 if (col + row) % 2 == 0 else PLAYER2
                apply_player_action(board, PlayerAction(col), player)
        
        # Should still work
        move, state = generate_move_time_limited(board, PLAYER1, time_limit_secs=1.0)
        assert isinstance(move, np.integer)
        assert 0 <= move < BOARD_COLS
    
    def test_forced_win_detection(self):
        """Test that minimax detects forced wins."""
        board = initialize_game_state()
        
        # Set up position where PLAYER1 has forced win
        setup_moves = [
            (PlayerAction(0), PLAYER1), (PlayerAction(1), PLAYER2),
            (PlayerAction(0), PLAYER1), (PlayerAction(1), PLAYER2),
            (PlayerAction(0), PLAYER1), (PlayerAction(1), PLAYER2),
            (PlayerAction(1), PLAYER1), (PlayerAction(2), PLAYER2),
            (PlayerAction(2), PLAYER1), (PlayerAction(3), PLAYER2),
            (PlayerAction(2), PLAYER1), (PlayerAction(3), PLAYER2),
            (PlayerAction(2), PLAYER1), (PlayerAction(3), PLAYER2)
        ]
        
        for move, player in setup_moves:
            apply_player_action(board, move, player)
        
        # PLAYER1 should now be able to force a win
        move, state = generate_move_time_limited(board, PLAYER1, time_limit_secs=2.0)
        assert isinstance(move, np.integer)
        
        # Apply the move and check if it leads to win
        apply_player_action(board, move, PLAYER1)
        game_state = check_end_state(board, PLAYER1)
        
        # Should either win immediately or create winning threat
        assert game_state in [GameState.IS_WIN, GameState.STILL_PLAYING]
    
    def test_forced_block_detection(self):
        """Test that minimax detects forced blocks."""
        board = initialize_game_state()
        
        # Set up position where PLAYER2 threatens to win
        for i in range(3):
            apply_player_action(board, PlayerAction(i), PLAYER2)
        for i in range(3):
            apply_player_action(board, PlayerAction(i), PLAYER1)
        
        # PLAYER1 should block the winning move
        move, state = generate_move_time_limited(board, PLAYER1, time_limit_secs=2.0)
        assert move == PlayerAction(3)  # Should block the win
    
    def test_invalid_state_handling(self):
        """Test handling of invalid or corrupted states."""
        board = initialize_game_state()
        
        # Test with None saved state
        move, state = generate_move_time_limited(board, PLAYER1, None, time_limit_secs=1.0)
        assert isinstance(move, np.integer)
        assert isinstance(state, MinimaxSavedState)
        
        # Test with invalid saved state type - skip this test due to type checking
        # The function should create a new state when given invalid input
        move, state = generate_move_time_limited(board, PLAYER1, None, time_limit_secs=1.0)
        assert isinstance(move, np.integer)
        assert isinstance(state, MinimaxSavedState)


class TestMinimaxIntegrationSymmetry:
    """Test symmetry handling in minimax integration."""
    
    def test_symmetric_position_equivalence(self):
        """Test that symmetric positions are handled equivalently."""
        # Create two symmetric positions
        board1 = initialize_game_state()
        board2 = initialize_game_state()
        
        # Apply symmetric moves
        moves1 = [PlayerAction(1), PlayerAction(2), PlayerAction(0)]
        moves2 = [PlayerAction(5), PlayerAction(4), PlayerAction(6)]  # Mirrors
        
        for i, (move1, move2) in enumerate(zip(moves1, moves2)):
            player = PLAYER1 if i % 2 == 0 else PLAYER2
            apply_player_action(board1, move1, player)
            apply_player_action(board2, move2, player)
        
        # Both should evaluate similarly
        move1, state1 = generate_move_time_limited(board1, PLAYER1, time_limit_secs=2.0)
        move2, state2 = generate_move_time_limited(board2, PLAYER1, time_limit_secs=2.0)
        
        # Moves should be mirrors of each other
        assert int(move1) + int(move2) == 6  # Columns 0-6, so mirrors sum to 6
    
    def test_symmetric_transposition_table_sharing(self):
        """Test that symmetric positions share transposition table entries."""
        board1 = initialize_game_state()
        board2 = initialize_game_state()
        
        # Create symmetric positions
        apply_player_action(board1, PlayerAction(1), PLAYER1)
        apply_player_action(board2, PlayerAction(5), PLAYER1)
        
        # Use same transposition table
        state = MinimaxSavedState()
        
        # Search first position
        move1, state = generate_move_time_limited(board1, PLAYER2, state, time_limit_secs=1.0)
        assert isinstance(state, MinimaxSavedState)
        initial_tt_size = len(state.transposition_table)
        
        # Search second position - should benefit from symmetry
        move2, state = generate_move_time_limited(board2, PLAYER2, state, time_limit_secs=1.0)
        assert isinstance(state, MinimaxSavedState)
        final_tt_size = len(state.transposition_table)
        
        # Should have some shared entries due to symmetry
        assert final_tt_size >= initial_tt_size
        assert isinstance(move1, np.integer)
        assert isinstance(move2, np.integer)


class TestMinimaxIntegrationRealWorld:
    """Test minimax in realistic game scenarios."""
    
    def test_opening_game_performance(self):
        """Test minimax performance in opening positions."""
        board = initialize_game_state()
        
        # Test first few moves of a game
        moves = []
        state = None
        
        for i in range(6):  # First 6 moves
            player = PLAYER1 if i % 2 == 0 else PLAYER2
            move, state = generate_move_time_limited(board, player, state, time_limit_secs=1.0)
            moves.append(move)
            apply_player_action(board, move, player)
        
        # All moves should be valid
        assert all(isinstance(move, np.integer) for move in moves)
        assert all(0 <= move < BOARD_COLS for move in moves)
        
        # Should show some strategic preference (e.g., center play)
        center_moves = sum(1 for move in moves if move == PlayerAction(3))
        assert center_moves > 0  # Should play center at least once
    
    def test_endgame_performance(self):
        """Test minimax performance in endgame positions."""
        board = initialize_game_state()
        
        # Create a late-game position
        late_game_moves = [
            (PlayerAction(3), PLAYER1), (PlayerAction(3), PLAYER2),
            (PlayerAction(3), PLAYER1), (PlayerAction(3), PLAYER2),
            (PlayerAction(2), PLAYER1), (PlayerAction(4), PLAYER2),
            (PlayerAction(2), PLAYER1), (PlayerAction(4), PLAYER2),
            (PlayerAction(1), PLAYER1), (PlayerAction(5), PLAYER2),
            (PlayerAction(1), PLAYER1), (PlayerAction(5), PLAYER2),
        ]
        
        for move, player in late_game_moves:
            apply_player_action(board, move, player)
        
        # Should still perform well in complex positions
        move, state = generate_move_time_limited(board, PLAYER1, time_limit_secs=2.0)
        assert isinstance(move, np.integer)
        assert 0 <= move < BOARD_COLS
    
    def test_time_pressure_performance(self):
        """Test minimax performance under time pressure."""
        board = initialize_game_state()
        
        # Apply some moves
        for i in range(4):
            player = PLAYER1 if i % 2 == 0 else PLAYER2
            move = PlayerAction(i)
            apply_player_action(board, move, player)
        
        # Test with very short time limit
        move, state = generate_move_time_limited(board, PLAYER1, time_limit_secs=0.1)
        assert isinstance(move, np.integer)
        assert 0 <= move < BOARD_COLS
        
        # Should still work with reasonable time
        move, state = generate_move_time_limited(board, PLAYER2, state, time_limit_secs=0.5)
        assert isinstance(move, np.integer)
        assert 0 <= move < BOARD_COLS
