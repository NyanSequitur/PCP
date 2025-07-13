"""
Behavioral contract tests for any Connect Four agent.

These tests define the expected behavior that ANY Connect Four agent should exhibit,
regardless of implementation details. They test the agent's interface and behavioral
contracts rather than internal implementation.
"""

import numpy as np
import pytest
import time
from game_utils import (
    initialize_game_state, apply_player_action, PlayerAction, PLAYER1, PLAYER2,
    check_end_state, GameState, BOARD_COLS, check_move_status, MoveStatus
)


@pytest.fixture(params=['minimax'])
def agent_factory(request):
    """Fixture that provides intelligent agent factories for testing."""
    if request.param == 'minimax':
        from agents.agent_minimax.minimax import create_minimax_agent
        def minimax_factory():
            """Create a minimax agent with test-appropriate parameters."""
            return create_minimax_agent(time_limit=2.0, max_depth=4)
        return minimax_factory
    else:
        raise ValueError(f"Unknown agent type: {request.param}")


@pytest.fixture
def random_agent_factory():
    """Fixture that provides random agent factory for basic testing."""
    from agents.agent_random.random import generate_move_random
    def random_factory():
        """Create a random agent for testing."""
        return generate_move_random, None
    return random_factory


class TestAgentBehavioralContracts:
    """Test behavioral contracts that any Connect Four agent should satisfy."""
    
    def test_agent_returns_valid_move_type(self, agent_factory):
        """Any agent should return a PlayerAction object."""
        agent_func, agent_state = agent_factory()
        board = initialize_game_state()
        
        move, new_state = agent_func(board, PLAYER1, agent_state)
        assert isinstance(move, (int, np.integer))
        assert 0 <= move < BOARD_COLS
    
    def test_agent_only_plays_valid_moves(self, agent_factory):
        """Any agent should only play legal moves."""
        agent_func, agent_state = agent_factory()
        board = initialize_game_state()
        
        # Test 100 random board positions
        for _ in range(100):
            move, agent_state = agent_func(board, PLAYER1, agent_state)
            move_status = check_move_status(board, move)
            assert move_status == MoveStatus.IS_VALID
            
            # Apply move to continue testing
            if move_status == MoveStatus.IS_VALID:
                apply_player_action(board, move, PLAYER1)
                
                # Check if game is over
                if check_end_state(board, PLAYER1) != GameState.STILL_PLAYING:
                    break
    
    def test_agent_plays_winning_move_when_available(self, agent_factory):
        """Any good agent should play a winning move when available."""
        agent_func, agent_state = agent_factory()
        
        # Test horizontal winning move
        board = initialize_game_state()
        # Create horizontal 3-in-a-row: X X X _ (columns 0,1,2, win at 3)
        apply_player_action(board, PlayerAction(0), PLAYER1)
        apply_player_action(board, PlayerAction(1), PLAYER1)
        apply_player_action(board, PlayerAction(2), PLAYER1)
        
        move, _ = agent_func(board, PLAYER1, agent_state)
        
        # Verify this is a winning move
        test_board = board.copy()
        apply_player_action(test_board, move, PLAYER1)
        assert check_end_state(test_board, PLAYER1) == GameState.IS_WIN
        
        # Test vertical winning move
        board = initialize_game_state()
        # Create vertical 3-in-a-row in column 3
        apply_player_action(board, PlayerAction(3), PLAYER1)
        apply_player_action(board, PlayerAction(3), PLAYER1)
        apply_player_action(board, PlayerAction(3), PLAYER1)
        
        move, _ = agent_func(board, PLAYER1, agent_state)
        
        # Verify this is a winning move
        test_board = board.copy()
        apply_player_action(test_board, move, PLAYER1)
        assert check_end_state(test_board, PLAYER1) == GameState.IS_WIN
    
    def test_agent_blocks_opponent_winning_move(self, agent_factory):
        """Any good agent should block opponent's winning move."""
        agent_func, agent_state = agent_factory()
        board = initialize_game_state()
        
        # Create a position where opponent threatens to win
        for col in range(3):
            apply_player_action(board, PlayerAction(col), PLAYER2)
        
        # Agent should block the win
        move, _ = agent_func(board, PLAYER1, agent_state)
        
        # Verify that not blocking would lose
        test_board = board.copy()
        apply_player_action(test_board, PlayerAction(3), PLAYER2)
        opponent_wins = check_end_state(test_board, PLAYER2) == GameState.IS_WIN
        
        if opponent_wins:
            # If opponent would win by playing column 3, agent should block
            assert move == PlayerAction(3)
    
    def test_agent_respects_time_limits(self, agent_factory):
        """Any agent should respect reasonable time limits."""
        agent_func, agent_state = agent_factory()
        board = initialize_game_state()
        
        start_time = time.time()
        move, _ = agent_func(board, PLAYER1, agent_state)
        elapsed = time.time() - start_time
        
        # Should not take more than 10 seconds for first move
        assert elapsed < 10.0
        assert isinstance(move, (int, np.integer))
    
    def test_agent_handles_nearly_full_board(self, agent_factory):
        """Any agent should handle nearly full board positions."""
        agent_func, agent_state = agent_factory()
        board = initialize_game_state()
        
        # Fill most columns, leaving only one open
        for col in range(BOARD_COLS - 1):
            for row in range(5):  # Fill 5 out of 6 rows
                player = PLAYER1 if (row + col) % 2 == 0 else PLAYER2
                apply_player_action(board, PlayerAction(col), player)
        
        # Agent should still be able to make a move
        move, _ = agent_func(board, PLAYER1, agent_state)
        assert isinstance(move, (int, np.integer))
        assert check_move_status(board, move) == MoveStatus.IS_VALID
    
    def test_agent_consistency_same_position(self, agent_factory):
        """Agent should make consistent moves from identical positions."""
        agent_func, agent_state = agent_factory()
        board = initialize_game_state()
        
        # Apply some moves to create a non-trivial position
        apply_player_action(board, PlayerAction(3), PLAYER1)
        apply_player_action(board, PlayerAction(3), PLAYER2)
        apply_player_action(board, PlayerAction(2), PLAYER1)
        
        # Agent should make the same move from this position
        move1, _ = agent_func(board, PLAYER1, agent_state)
        move2, _ = agent_func(board, PLAYER1, agent_state)
        
        assert move1 == move2


# Test individual agents
def test_minimax_behavioral_contracts():
    """Test minimax agent behavioral contracts."""
    from agents.agent_minimax.minimax import create_minimax_agent
    
    test_instance = TestAgentBehavioralContracts()
    agent_func, agent_state = create_minimax_agent(time_limit=2.0, max_depth=4)
    
    # Create a simple factory function for this agent
    def agent_factory():
        """Create a minimax agent instance for testing."""
        return agent_func, agent_state
    
    # Run key behavioral tests
    test_instance.test_agent_returns_valid_move_type(agent_factory)
    test_instance.test_agent_respects_time_limits(agent_factory)


def test_random_behavioral_contracts():
    """Test random agent behavioral contracts."""
    from agents.agent_random.random import generate_move_random
    
    test_instance = TestAgentBehavioralContracts()
    
    def agent_factory():
        """Create a random agent instance for testing."""
        return generate_move_random, None
    
    # Run key behavioral tests
    test_instance.test_agent_returns_valid_move_type(agent_factory)
    test_instance.test_agent_respects_time_limits(agent_factory)


class TestRandomAgentBasics:
    """Basic tests for non-strategic agents like random agent."""
    
    def test_random_agent_returns_valid_move_type(self, random_agent_factory):
        """Random agent should return valid move types."""
        agent_func, agent_state = random_agent_factory()
        board = initialize_game_state()
        
        move, new_state = agent_func(board, PLAYER1, agent_state)
        assert isinstance(move, (int, np.integer))
        assert 0 <= move < BOARD_COLS
    
    def test_random_agent_only_plays_valid_moves(self, random_agent_factory):
        """Random agent should only play legal moves."""
        agent_func, agent_state = random_agent_factory()
        board = initialize_game_state()
        
        # Test 50 random board positions
        for _ in range(50):
            move, agent_state = agent_func(board, PLAYER1, agent_state)
            move_status = check_move_status(board, move)
            assert move_status == MoveStatus.IS_VALID
            
            # Apply move to continue testing
            if move_status == MoveStatus.IS_VALID:
                apply_player_action(board, move, PLAYER1)
                
                # Check if game is over
                if check_end_state(board, PLAYER1) != GameState.STILL_PLAYING:
                    break
    
    def test_random_agent_respects_time_limits(self, random_agent_factory):
        """Random agent should respect reasonable time limits."""
        agent_func, agent_state = random_agent_factory()
        board = initialize_game_state()
        
        start_time = time.time()
        move, _ = agent_func(board, PLAYER1, agent_state)
        elapsed = time.time() - start_time
        
        # Should be very fast for random agent
        assert elapsed < 1.0
        assert isinstance(move, (int, np.integer))
    
    def test_random_agent_handles_nearly_full_board(self, random_agent_factory):
        """Random agent should handle nearly full board positions."""
        agent_func, agent_state = random_agent_factory()
        board = initialize_game_state()
        
        # Fill most columns, leaving only one open
        for col in range(BOARD_COLS - 1):
            for row in range(5):  # Fill 5 out of 6 rows
                player = PLAYER1 if (row + col) % 2 == 0 else PLAYER2
                apply_player_action(board, PlayerAction(col), player)
        
        # Agent should still be able to make a move
        move, _ = agent_func(board, PLAYER1, agent_state)
        assert isinstance(move, (int, np.integer))
        assert check_move_status(board, move) == MoveStatus.IS_VALID
