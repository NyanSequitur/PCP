"""
Heuristic evaluation functions for minimax agent.

This module contains all the heuristic evaluation functions used to score
board positions when the search cannot continue to terminal states.
"""

import numpy as np
from typing import List

from game_utils import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, BOARD_COLS, BOARD_ROWS


def get_other_player(player: BoardPiece) -> BoardPiece:
    """
    Return the opponent of the given player.

    Parameters
    ----------
    player : BoardPiece
        The current player.

    Returns
    -------
    BoardPiece
        The opponent player.
    """
    return PLAYER1 if player == PLAYER2 else PLAYER2


def evaluate_board(board: np.ndarray, player: BoardPiece) -> float:
    """
    Enhanced heuristic evaluation of the board for the given player.
    
    The evaluation considers multiple factors:
    - Pattern-based scoring (windows of 4 cells)
    - Positional weights (center preference)
    - Multiple threat detection
    - Piece connectivity
    - Center column control

    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to evaluate for.

    Returns
    -------
    float
        Enhanced heuristic score.
    """
    score = 0.0
    
    # 1. Traditional window-based scoring (enhanced)
    score += _score_horizontal_windows(board, player)
    score += _score_vertical_windows(board, player)
    score += _score_diagonal_windows(board, player)
    
    # 2. Enhanced center column control
    score += _score_center_column(board, player)
    
    # 3. Positional weights
    score += _score_positional_weights(board, player)
    
    # 4. Multiple threat detection
    score += _detect_multiple_threats(board, player)
    score -= _detect_multiple_threats(board, get_other_player(player))
    
    # 5. Connectivity bonus
    score += _score_connectivity(board, player)
    score -= _score_connectivity(board, get_other_player(player)) * 0.8
    
    return score


def _score_window(window: np.ndarray, player: BoardPiece) -> float:
    """
    Enhanced scoring for a 4-cell window with better threat detection.
    
    Parameters
    ----------
    window : np.ndarray
        A 4-element array representing a window on the board.
    player : BoardPiece
        The player to score for.
        
    Returns
    -------
    float
        Score for this window.
    """
    player_count = np.count_nonzero(window == player)
    opponent_count = np.count_nonzero(window == get_other_player(player))
    empty_count = np.count_nonzero(window == NO_PLAYER)
    
    # Terminal states
    if player_count == 4:
        return 1000.0  # Win - much higher value
    if opponent_count == 4:
        return -1000.0  # Opponent win
    
    # Strong threats
    if player_count == 3 and empty_count == 1:
        return 50.0   # Strong threat
    if opponent_count == 3 and empty_count == 1:
        return -45.0  # Must block
    
    # Medium threats  
    if player_count == 2 and empty_count == 2:
        # Check if pieces are connected (more valuable)
        if _are_pieces_connected(window, player):
            return 10.0  # Connected pair
        else:
            return 6.0   # Separated pair
    if opponent_count == 2 and empty_count == 2:
        if _are_pieces_connected(window, get_other_player(player)):
            return -8.0  # Block connected pair
        else:
            return -4.0  # Block separated pair
    
    # Single pieces (for early game positioning)
    if player_count == 1 and empty_count == 3:
        return 2.0
    if opponent_count == 1 and empty_count == 3:
        return -1.0
    
    return 0.0


def _are_pieces_connected(window: np.ndarray, player: BoardPiece) -> bool:
    """
    Check if player's pieces in the window are connected (adjacent).
    
    Parameters
    ----------
    window : np.ndarray
        A 4-element window.
    player : BoardPiece
        The player to check.
        
    Returns
    -------
    bool
        True if pieces are connected.
    """
    player_positions = [i for i, piece in enumerate(window) if piece == player]
    if len(player_positions) < 2:
        return True  # Single piece or no pieces are trivially "connected"
    
    # Check if positions are consecutive
    for i in range(len(player_positions) - 1):
        if player_positions[i + 1] - player_positions[i] != 1:
            return False
    return True


def _score_horizontal_windows(board: np.ndarray, player: BoardPiece) -> float:
    """
    Score all horizontal 4-cell windows.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to score for.
        
    Returns
    -------
    float
        Total score for horizontal windows.
    """
    score = 0.0
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS - 3):
            window = board[row, col:col+4]
            score += _score_window(window, player)
    return score


def _score_vertical_windows(board: np.ndarray, player: BoardPiece) -> float:
    """
    Score all vertical 4-cell windows.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to score for.
        
    Returns
    -------
    float
        Total score for vertical windows.
    """
    score = 0.0
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS - 3):
            window = board[row:row+4, col]
            score += _score_window(window, player)
    return score


def _score_diagonal_windows(board: np.ndarray, player: BoardPiece) -> float:
    """
    Score all diagonal 4-cell windows (both directions).
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to score for.
        
    Returns
    -------
    float
        Total score for diagonal windows.
    """
    score = 0.0
    
    # Positive diagonal (↗︎)
    for row in range(BOARD_ROWS - 3):
        for col in range(BOARD_COLS - 3):
            window = np.array([board[row+i, col+i] for i in range(4)])
            score += _score_window(window, player)
    
    # Negative diagonal (↘︎)
    for row in range(BOARD_ROWS - 3):
        for col in range(3, BOARD_COLS):
            window = np.array([board[row+i, col-i] for i in range(4)])
            score += _score_window(window, player)
    
    return score


def _score_center_column(board: np.ndarray, player: BoardPiece) -> float:
    """
    Enhanced center column scoring with positional weights.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to score for.
        
    Returns
    -------
    float
        Score for center column control.
    """
    center_col = BOARD_COLS // 2
    score = 0.0
    
    # Weight center pieces by their height (lower pieces are more valuable)
    for row in range(BOARD_ROWS):
        if board[row, center_col] == player:
            height_weight = BOARD_ROWS - row  # Higher weight for lower positions
            score += 4.0 * height_weight
        elif board[row, center_col] == get_other_player(player):
            height_weight = BOARD_ROWS - row
            score -= 3.0 * height_weight
    
    return score


def _score_positional_weights(board: np.ndarray, player: BoardPiece) -> float:
    """
    Score based on positional weights - center columns are more valuable.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to score for.
        
    Returns
    -------
    float
        Positional score.
    """
    # Positional weights for each column (center columns more valuable)
    column_weights = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]
    
    score = 0.0
    for col in range(BOARD_COLS):
        col_weight = column_weights[col]
        for row in range(BOARD_ROWS):
            if board[row, col] == player:
                # Lower pieces are more valuable (harder to block)
                height_bonus = (BOARD_ROWS - row) * 0.5
                score += col_weight + height_bonus
            elif board[row, col] == get_other_player(player):
                height_bonus = (BOARD_ROWS - row) * 0.5
                score -= (col_weight + height_bonus) * 0.8
    
    return score


def _detect_multiple_threats(board: np.ndarray, player: BoardPiece) -> float:
    """
    Detect and score positions that create multiple winning threats.
    
    A threat is a position where a player has 3 pieces in a row with one empty space
    that would complete a winning line of 4.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to score for.
        
    Returns
    -------
    float
        Score for multiple threats.
    """
    threats = 0
    
    # Check all possible 4-in-a-row windows for potential threats
    # Horizontal threats
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS - 3):
            window = board[row, col:col+4]
            if np.count_nonzero(window == player) == 3 and np.count_nonzero(window == NO_PLAYER) == 1:
                threats += 1
    
    # Vertical threats
    for col in range(BOARD_COLS):
        for row in range(BOARD_ROWS - 3):
            window = board[row:row+4, col]
            if np.count_nonzero(window == player) == 3 and np.count_nonzero(window == NO_PLAYER) == 1:
                threats += 1
    
    # Diagonal threats (both directions)
    for row in range(BOARD_ROWS - 3):
        for col in range(BOARD_COLS - 3):
            # Positive diagonal
            window = np.array([board[row+i, col+i] for i in range(4)])
            if np.count_nonzero(window == player) == 3 and np.count_nonzero(window == NO_PLAYER) == 1:
                threats += 1
            
            # Negative diagonal
            if col >= 3:
                window = np.array([board[row+i, col-i] for i in range(4)])
                if np.count_nonzero(window == player) == 3 and np.count_nonzero(window == NO_PLAYER) == 1:
                    threats += 1
    
    # Multiple threats are exponentially more valuable
    if threats >= 2:
        return 100.0 * (threats - 1)  # Bonus for multiple threats
    return 0.0


def _score_connectivity(board: np.ndarray, player: BoardPiece) -> float:
    """
    Score based on piece connectivity and potential for future connections.
    
    This function rewards positions where the player's pieces are well-connected,
    as connected pieces are more likely to form winning combinations.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to score for.
        
    Returns
    -------
    float
        Connectivity score.
    """
    score = 0.0
    
    # Check for connected pieces (adjacent horizontally, vertically, or diagonally)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row, col] == player:
                # Count connected pieces in all 8 directions
                directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
                connections = 0
                
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    if (0 <= new_row < BOARD_ROWS and 0 <= new_col < BOARD_COLS and
                        board[new_row, new_col] == player):
                        connections += 1
                
                # Bonus for well-connected pieces
                score += connections * 1.5
    
    return score
