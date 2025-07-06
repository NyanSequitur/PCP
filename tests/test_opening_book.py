"""
Comprehensive tests for the opening book implementation in the minimax agent.
Tests the board encoding, opening book loading, and position lookup functionality.
"""

import pytest
import numpy as np
import math
import io
import csv
import zipfile
from unittest.mock import Mock, patch, MagicMock
from agents.agent_minimax.minimax import MinimaxSavedState
from game_utils import BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, apply_player_action


class TestOpeningBook:
    """Test suite for opening book functionality."""
    
    def test_board_to_book_key_empty_board(self):
        """Test encoding an empty board."""
        state = MinimaxSavedState()
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        key = state._board_to_book_key(board)
        # Empty board should be all 'b' (42 squares)
        expected = "b" * (BOARD_ROWS * BOARD_COLS)
        assert key == expected
        assert len(key) == 42
    
    def test_board_to_book_key_column_major_order(self):
        """Test that board encoding follows column-major, bottom-to-top order."""
        state = MinimaxSavedState()
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        
        # Place a piece in column 0, row 0 (bottom-left)
        board[0, 0] = PLAYER1
        key = state._board_to_book_key(board)
        
        # In column-major bottom-to-top order, this should be the first character
        assert key[0] == 'x'  # PLAYER1 -> 'x'
        assert key[1:6] == 'bbbbb'  # rest of column 0 empty
        assert key[6:] == 'b' * 36  # rest of board empty
    
    def test_board_to_book_key_player_encoding(self):
        """Test correct player piece encoding."""
        state = MinimaxSavedState()
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        
        # Place different pieces
        board[0, 0] = PLAYER1  # 'x'
        board[1, 0] = PLAYER2  # 'o'
        board[2, 0] = NO_PLAYER  # 'b'
        
        key = state._board_to_book_key(board)
        
        # Check encoding in column-major, bottom-to-top order
        assert key[0] == 'x'  # PLAYER1 at [0,0]
        assert key[1] == 'o'  # PLAYER2 at [1,0]
        assert key[2] == 'b'  # NO_PLAYER at [2,0]
    
    def test_board_to_book_key_complex_position(self):
        """Test encoding a more complex board position."""
        state = MinimaxSavedState()
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        
        # Create a specific pattern
        # Column 0: X, O, X (bottom to top)
        board[0, 0] = PLAYER1
        board[1, 0] = PLAYER2
        board[2, 0] = PLAYER1
        
        # Column 1: O, X (bottom to top)
        board[0, 1] = PLAYER2
        board[1, 1] = PLAYER1
        
        key = state._board_to_book_key(board)
        
        # Column 0 (bottom to top): X, O, X, empty, empty, empty
        assert key[0:6] == 'xoxbbb'
        # Column 1 (bottom to top): O, X, empty, empty, empty, empty
        assert key[6:12] == 'oxbbbb'
        # Remaining columns should be empty
        assert key[12:] == 'b' * 30
    
    def test_load_opening_book_already_loaded(self):
        """Test that loading is skipped if already loaded."""
        state = MinimaxSavedState()
        state.opening_book_loaded = True
        
        # Mock the URL opening to ensure it's not called
        with patch('urllib.request.urlopen') as mock_urlopen:
            state._load_opening_book()
            mock_urlopen.assert_not_called()
    
    def test_load_opening_book_already_failed(self):
        """Test that loading is skipped if previously failed."""
        state = MinimaxSavedState()
        state.opening_book_failed = True
        
        # Mock the URL opening to ensure it's not called
        with patch('urllib.request.urlopen') as mock_urlopen:
            state._load_opening_book()
            mock_urlopen.assert_not_called()
    
    def test_load_opening_book_network_success(self):
        """Test successful opening book loading from network."""
        state = MinimaxSavedState()
        
        # Create mock CSV data
        csv_data = "x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win\n"
        csv_data += "o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss\n"
        csv_data += "b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,draw\n"
        
        # Create mock zip file
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ['connect-4.csv']
        mock_zip.read.return_value = csv_data.encode('utf-8')
        
        with patch('agents.agent_minimax.minimax.urllib.request.urlopen') as mock_urlopen, \
             patch('agents.agent_minimax.minimax.zipfile.ZipFile', return_value=mock_zip):
            
            mock_response = MagicMock()
            mock_response.read.return_value = b'mock_zip_data'
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            state._load_opening_book()
            
            # Check that the book was loaded
            assert state.opening_book_loaded
            assert not state.opening_book_failed
            assert len(state.opening_book) == 3
            
            # Check specific entries
            key1 = "x" + "o" + "b" * 40
            key2 = "o" + "x" + "b" * 40
            key3 = "b" * 42
            
            assert state.opening_book[key1] == math.inf  # win
            assert state.opening_book[key2] == -math.inf  # loss
            assert state.opening_book[key3] == 0.0  # draw
    
    def test_load_opening_book_network_failure_no_fallback(self):
        """Test opening book loading failure when network fails and no fallback."""
        state = MinimaxSavedState()
        
        from urllib.error import URLError
        
        with patch('agents.agent_minimax.minimax.urllib.request.urlopen', side_effect=URLError("Network error")), \
             patch('agents.agent_minimax.minimax.zipfile.ZipFile', side_effect=Exception("File not found")):
            
            state._load_opening_book()
            
            # Should mark as failed
            assert state.opening_book_failed
            assert not state.opening_book_loaded
            assert len(state.opening_book) == 0
    
    def test_load_opening_book_invalid_csv(self):
        """Test handling of invalid CSV data."""
        state = MinimaxSavedState()
        
        # Create invalid CSV data (wrong number of columns)
        csv_data = "x,o,b,win\n"  # Only 4 columns instead of 43
        csv_data += "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,win\n"  # Valid row
        
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ['connect-4.csv']
        mock_zip.read.return_value = csv_data.encode('utf-8')
        
        with patch('agents.agent_minimax.minimax.urllib.request.urlopen') as mock_urlopen, \
             patch('agents.agent_minimax.minimax.zipfile.ZipFile', return_value=mock_zip):
            
            mock_response = MagicMock()
            mock_response.read.return_value = b'mock_zip_data'
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            state._load_opening_book()
            
            # Should load successfully but skip invalid rows
            assert state.opening_book_loaded
            assert not state.opening_book_failed
            assert len(state.opening_book) == 1  # Only the valid row
    
    def test_lookup_opening_book_not_loaded(self):
        """Test opening book lookup when not loaded."""
        state = MinimaxSavedState()
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        
        # Mock successful loading
        with patch.object(state, '_load_opening_book') as mock_load:
            # Set up the book state after loading
            def setup_book():
                state.opening_book = {"test_key": 5.0}
                state.opening_book_loaded = True
            
            mock_load.side_effect = setup_book
            
            # Mock board_to_book_key to return our test key
            with patch.object(state, '_board_to_book_key', return_value="test_key"):
                result = state.lookup_opening_book(board)
                assert result == 5.0
                mock_load.assert_called_once()
    
    def test_lookup_opening_book_failed_load(self):
        """Test opening book lookup when loading failed."""
        state = MinimaxSavedState()
        state.opening_book_failed = True
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        
        result = state.lookup_opening_book(board)
        assert result is None
    
    def test_lookup_opening_book_position_found(self):
        """Test successful position lookup."""
        state = MinimaxSavedState()
        state.opening_book_loaded = True
        state.opening_book = {"test_position": 10.0}
        
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        
        with patch.object(state, '_board_to_book_key', return_value="test_position"):
            result = state.lookup_opening_book(board)
            assert result == 10.0
    
    def test_lookup_opening_book_position_not_found(self):
        """Test position lookup when position not in book."""
        state = MinimaxSavedState()
        state.opening_book_loaded = True
        state.opening_book = {"other_position": 10.0}
        
        board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        
        with patch.object(state, '_board_to_book_key', return_value="unknown_position"):
            result = state.lookup_opening_book(board)
            assert result is None
    
    def test_opening_book_integration_with_real_board(self):
        """Test opening book with actual Connect Four positions."""
        state = MinimaxSavedState()
        
        # Set up a mock opening book with a real position
        empty_board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        empty_key = state._board_to_book_key(empty_board)
        
        # Simulate having the empty board in the opening book
        state.opening_book = {empty_key: 0.0}  # Draw value
        state.opening_book_loaded = True
        
        # Test lookup
        result = state.lookup_opening_book(empty_board)
        assert result == 0.0
        
        # Test with a different position
        board_with_move = empty_board.copy()
        apply_player_action(board_with_move, PlayerAction(3), PLAYER1)
        
        # This position shouldn't be in our mock book
        result = state.lookup_opening_book(board_with_move)
        assert result is None
    
    def test_outcome_value_mapping(self):
        """Test correct mapping of outcome strings to values."""
        state = MinimaxSavedState()
        
        # Test data with all possible outcomes
        csv_data = "x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win\n"
        csv_data += "o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss\n"
        csv_data += "b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,draw\n"
        
        mock_zip = MagicMock()
        mock_zip.namelist.return_value = ['connect-4.csv']
        mock_zip.read.return_value = csv_data.encode('utf-8')
        
        with patch('agents.agent_minimax.minimax.urllib.request.urlopen') as mock_urlopen, \
             patch('agents.agent_minimax.minimax.zipfile.ZipFile', return_value=mock_zip):
            
            mock_response = MagicMock()
            mock_response.read.return_value = b'mock_zip_data'
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            state._load_opening_book()
            
            # Check that outcomes are mapped correctly
            win_key = "x" + "o" + "b" * 40
            loss_key = "o" + "x" + "b" * 40
            draw_key = "b" * 2 + "x" + "b" * 39
            
            assert state.opening_book[win_key] == math.inf
            assert state.opening_book[loss_key] == -math.inf
            assert state.opening_book[draw_key] == 0.0


def test_opening_book_performance():
    """Test opening book performance characteristics."""
    state = MinimaxSavedState()
    
    # Create a large mock opening book
    large_book = {}
    for i in range(1000):
        key = f"position_{i:03d}" + "b" * 31  # 42 chars total
        large_book[key] = float(i % 3 - 1)  # -1, 0, or 1
    
    state.opening_book = large_book
    state.opening_book_loaded = True
    
    # Test lookup performance
    board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
    
    import time
    start_time = time.time()
    
    # Perform many lookups
    for _ in range(1000):
        state.lookup_opening_book(board)
    
    end_time = time.time()
    
    # Should be very fast (less than 0.1 seconds for 1000 lookups)
    assert end_time - start_time < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
