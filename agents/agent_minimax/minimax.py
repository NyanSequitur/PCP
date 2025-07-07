"""
Minimax agent for Connect Four with iterative deepening and alpha-beta pruning.

Implements a time-limited move generator using Negamax with alpha-beta pruning and a simple heuristic.
"""

import time
import math
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from game_utils import (
    BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2,
    apply_player_action, check_end_state, GameState,
    check_move_status, MoveStatus, PlayerAction, BoardPiece, SavedState
)

# Remark: Awesome! There really isn't much for me to complain about, great work!
# Take the remarks I left as suggestions, maybe you find them helpful.

@dataclass
class TranspositionEntry:
    """Entry in the transposition table."""
    value: float
    depth: int
    best_move: Optional[PlayerAction]
    flag: str  # 'exact', 'lower', 'upper' for exact value, lower bound, upper bound

class MinimaxSavedState(SavedState):
    """Saved state for minimax agent with transposition table."""
    
    def __init__(self, max_table_size: int = 1000000):
        self.transposition_table: Dict[str, TranspositionEntry] = {}
        self.move_ordering: Dict[str, list[PlayerAction]] = {}
        self.max_table_size = max_table_size
    
    def _cleanup_table_if_needed(self):
        """Clean up transposition table if it gets too large."""
        if len(self.transposition_table) > self.max_table_size:
            # Remove oldest entries (simple cleanup strategy)
            # In a more sophisticated implementation, we could use LRU or depth-based cleanup
            items_to_remove = len(self.transposition_table) - self.max_table_size // 2
            keys_to_remove = list(self.transposition_table.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.transposition_table[key]
    
    def get_board_hash(self, board: np.ndarray) -> str:
        """Get a hash string for the board state, using canonical form for symmetry."""
        canonical_board = self._get_canonical_board(board)
        return canonical_board.tobytes().hex()
    
    def _get_canonical_board(self, board: np.ndarray) -> np.ndarray:
        """
        Get the canonical form of the board by choosing the lexicographically smaller
        representation between the original and its horizontal mirror.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        np.ndarray
            The canonical board representation.
        """
        # Mirror the board horizontally (flip left-right)
        mirrored = np.fliplr(board)
        
        # Compare boards lexicographically to determine canonical form
        # We flatten and compare as 1D arrays for efficiency
        original_flat = board.flatten()
        mirrored_flat = mirrored.flatten()
        
        # Return the lexicographically smaller one
        for i in range(len(original_flat)):
            if original_flat[i] < mirrored_flat[i]:
                return board
            elif original_flat[i] > mirrored_flat[i]:
                return mirrored
        
        # If they're equal, return the original
        return board
    
    def _is_board_mirrored(self, board: np.ndarray) -> bool:
        """
        Check if the canonical form is the mirrored version of the original board.
        
        Parameters
        ----------
        board : np.ndarray
            The game board.
            
        Returns
        -------
        bool
            True if the canonical form is the mirrored version.
        """
        # More efficient: directly compare with mirrored version
        mirrored = np.fliplr(board)
        original_flat = board.flatten()
        mirrored_flat = mirrored.flatten()
        
        # Check if mirrored version is lexicographically smaller
        for i in range(len(original_flat)):
            if original_flat[i] < mirrored_flat[i]:
                return False  # Original is smaller, not mirrored
            elif original_flat[i] > mirrored_flat[i]:
                return True   # Mirrored is smaller, so canonical is mirrored
        
        # If they're equal, canonical is original (not mirrored)
        return False
    
    def _mirror_move(self, move: PlayerAction) -> PlayerAction:
        """
        Mirror a move horizontally.
        
        Parameters
        ----------
        move : PlayerAction
            The original move.
            
        Returns
        -------
        PlayerAction
            The mirrored move.
        """
        return PlayerAction(BOARD_COLS - 1 - int(move))
    
    def store_position(self, board: np.ndarray, depth: int, value: float, 
                      best_move: Optional[PlayerAction], flag: str):
        """Store a position in the transposition table, handling symmetry."""
        board_hash = self.get_board_hash(board)
        
        # If the canonical form is mirrored, mirror the best move before storing
        if best_move is not None and self._is_board_mirrored(board):
            best_move = self._mirror_move(best_move)
        
        self.transposition_table[board_hash] = TranspositionEntry(
            value=value, depth=depth, best_move=best_move, flag=flag
        )
        
        # Clean up table if it gets too large
        self._cleanup_table_if_needed()
    
    def lookup_position(self, board: np.ndarray, depth: int, alpha: float, beta: float) -> Tuple[bool, float]:
        """Look up a position in the transposition table, handling symmetry."""
        board_hash = self.get_board_hash(board)
        if board_hash not in self.transposition_table:
            return False, 0.0
        
        entry = self.transposition_table[board_hash]
        
        # Only use entries with sufficient depth
        if entry.depth < depth:
            return False, 0.0
        
        # Check bounds and return appropriate values
        if entry.flag == 'exact':
            return True, entry.value
        elif entry.flag == 'lower' and entry.value >= beta:
            return True, entry.value
        elif entry.flag == 'upper' and entry.value <= alpha:
            return True, entry.value
        
        return False, 0.0
    
    def get_best_move(self, board: np.ndarray) -> Optional[PlayerAction]:
        """Get the best move for a position if available, handling symmetry."""
        board_hash = self.get_board_hash(board)
        if board_hash in self.transposition_table:
            best_move = self.transposition_table[board_hash].best_move
            if best_move is not None:
                # If the canonical form is mirrored, mirror the move back
                if self._is_board_mirrored(board):
                    return self._mirror_move(best_move)
                else:
                    return best_move
        return None
    
    def get_move_ordering(self, board: np.ndarray) -> list[PlayerAction]:
        """Get move ordering for a position."""
        board_hash = self.get_board_hash(board)
        if board_hash in self.move_ordering:
            return self.move_ordering[board_hash]
        return []
    
    def store_move_ordering(self, board: np.ndarray, moves: list[PlayerAction]):
        """Store move ordering for a position."""
        board_hash = self.get_board_hash(board)
        self.move_ordering[board_hash] = moves

def _get_valid_columns(board: np.ndarray) -> list[int]:
    """
    Get all valid columns for move placement.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
        
    Returns
    -------
    list[int]
        List of valid column indices.
    """
    return [c for c in range(BOARD_COLS) 
            if check_move_status(board, PlayerAction(c)) == MoveStatus.IS_VALID]


def _order_moves(board: np.ndarray, valid_cols: list[int], 
                 state: MinimaxSavedState) -> list[int]:
    """
    Order moves for better alpha-beta pruning efficiency.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    valid_cols : list[int]
        List of valid column indices.
    state : MinimaxSavedState
        The saved state with transposition table.
        
    Returns
    -------
    list[int]
        Ordered list of column indices.
    """
    ordered_moves = []
    remaining_cols = valid_cols.copy()
    
    # Try transposition table best move first
    tt_best_move = state.get_best_move(board)
    if tt_best_move is not None and int(tt_best_move) in remaining_cols:
        ordered_moves.append(int(tt_best_move))
        remaining_cols.remove(int(tt_best_move))
    
    # Add center columns first (better move ordering for Connect Four)
    center = BOARD_COLS // 2
    for offset in range(BOARD_COLS):
        if not remaining_cols:  # No more columns to add
            break
        
        # Try center column first
        if offset == 0:
            if center in remaining_cols:
                ordered_moves.append(center)
                remaining_cols.remove(center)
        else:
            # Try columns to the right and left of center
            for direction in [1, -1]:
                col = center + direction * offset
                if 0 <= col < BOARD_COLS and col in remaining_cols:
                    ordered_moves.append(col)
                    remaining_cols.remove(col)
                    break
    
    # Add any remaining columns (should not happen with correct logic)
    ordered_moves.extend(remaining_cols)
    return ordered_moves


def _get_transposition_flag(value: float, original_alpha: float, beta: float) -> str:
    """
    Determine the transposition table flag based on alpha-beta bounds.
    
    Parameters
    ----------
    value : float
        The evaluated value.
    original_alpha : float
        The original alpha value before search.
    beta : float
        The beta value.
        
    Returns
    -------
    str
        The flag: 'exact', 'upper', or 'lower'.
    """
    if value <= original_alpha:
        return 'upper'
    elif value >= beta:
        return 'lower'
    else:
        return 'exact'


def _negamax(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    turn: BoardPiece,
    state: MinimaxSavedState,
    deadline: float
) -> float:
    """
    Negamax search with alpha-beta pruning and transposition table.

    Parameters
    ----------
    board : np.ndarray
        Board state.
    depth : int
        Search depth.
    alpha : float
        Alpha value.
    beta : float
        Beta value.
    turn : BoardPiece
        Player to move.
    state : MinimaxSavedState
        The saved state with transposition table.
    deadline : float
        Time deadline for search.

    Returns
    -------
    float
        Heuristic value of the board.
    """
    if time.monotonic() > deadline:
        raise TimeoutError  # Stop search if time limit exceeded
    
    # Check transposition table first
    found, tt_value = state.lookup_position(board, depth, alpha, beta)
    if found:
        return tt_value
    
    original_alpha = alpha
    
    # Check for terminal states
    previous_player = _other(turn)
    game_state = check_end_state(board, previous_player)
    if game_state == GameState.IS_WIN:
        return -math.inf  # Previous player just won
    if game_state == GameState.IS_DRAW:
        return 0.0
    if depth == 0:
        return _heuristic(board, turn)
    
    # Search all moves
    value = -math.inf
    best_move = None
    valid_cols = _get_valid_columns(board)
    
    # Handle case where no valid moves exist (should not happen in normal game)
    if not valid_cols:
        return 0.0  # Return neutral score if no moves available
    
    ordered_moves = _order_moves(board, valid_cols, state)
    
    for col in ordered_moves:
        if time.monotonic() > deadline:
            raise TimeoutError
        
        new_board = board.copy()
        apply_player_action(new_board, PlayerAction(col), turn)
        
        # Negamax recursion: switch player and invert score
        score = -_negamax(new_board, depth - 1, -beta, -alpha, _other(turn), state, deadline)
        
        if score > value:
            value = score
            best_move = PlayerAction(col)
        
        alpha = max(alpha, score)
        if alpha >= beta:
            break  # Beta cutoff
    
    # Store in transposition table
    flag = _get_transposition_flag(value, original_alpha, beta)
    state.store_position(board, depth, value, best_move, flag)
    
    return value


def _search_depth(
    board: np.ndarray,
    player: BoardPiece,
    depth: int,
    valid_cols: list[int],
    state: MinimaxSavedState,
    deadline: float
) -> Tuple[int, float]:
    """
    Search at a specific depth and return the best move and score.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to move.
    depth : int
        Search depth.
    valid_cols : list[int]
        List of valid column indices.
    state : MinimaxSavedState
        The saved state with transposition table.
    deadline : float
        Time deadline for search.
        
    Returns
    -------
    Tuple[int, float]
        Best column and its score.
    """
    alpha = -math.inf
    beta = math.inf
    best_score = -math.inf
    best_col = valid_cols[0]
    
    # Order moves for this depth
    ordered_cols = _order_moves(board, valid_cols, state)
    
    for col in ordered_cols:
        if time.monotonic() > deadline:
            raise TimeoutError
        
        trial_board = board.copy()
        apply_player_action(trial_board, PlayerAction(col), player)
        
        # Negamax search for opponent's best response
        score = -_negamax(trial_board, depth - 1, -beta, -alpha, _other(player), state, deadline)
        
        if score > best_score:
            best_score = score
            best_col = col
        
        alpha = max(alpha, score)
    
    return best_col, best_score


def _initialize_search(
    board: np.ndarray,
    saved_state: Optional[SavedState]
) -> Tuple[MinimaxSavedState, list[int], int]:
    """
    Initialize the search state and determine starting move.
    
    Parameters
    ----------
    board : np.ndarray
        The game board.
    saved_state : Optional[SavedState]
        The saved state from previous moves.
        
    Returns
    -------
    Tuple[MinimaxSavedState, list[int], int]
        Initialized state, valid columns, and starting best column.
    """
    # Initialize or reuse saved state
    if saved_state is None or not isinstance(saved_state, MinimaxSavedState):
        state = MinimaxSavedState()
    else:
        state = saved_state
    
    valid_cols = _get_valid_columns(board)
    if not valid_cols:
        raise Exception("No valid moves available.")
    
    # Check if we have a best move from previous search
    best_col = state.get_best_move(board)
    if best_col is None or int(best_col) not in valid_cols:
        best_col = valid_cols[0]
    else:
        best_col = int(best_col)
    
    return state, valid_cols, best_col


def generate_move_time_limited(
    board: np.ndarray,
    player: BoardPiece,
    saved_state: Optional[SavedState] = None,
    time_limit_secs: float = 5.0,
    max_depth: int = 20
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generate a move for the given player using iterative deepening Negamax with alpha-beta pruning.

    Parameters
    ----------
    board : np.ndarray
        The current game board.
    player : BoardPiece
        The player to move.
    saved_state : Optional[SavedState], optional
        State to persist between moves (default is None).
    time_limit_secs : float, optional
        Time limit for move search in seconds (default is 5.0).
    max_depth : int, optional
        Maximum search depth (default is 20).

    Returns
    -------
    Tuple[PlayerAction, Optional[SavedState]]
        The chosen move and updated state.
    """
    # Input validation
    if time_limit_secs <= 0:
        raise ValueError("Time limit must be positive")
    if max_depth <= 0:
        raise ValueError("Maximum depth must be positive")
    if player not in [PLAYER1, PLAYER2]:
        raise ValueError("Player must be PLAYER1 or PLAYER2")
    
    start_time = time.monotonic()
    deadline = start_time + time_limit_secs
    
    # Initialize search
    state, valid_cols, best_col = _initialize_search(board, saved_state)
    best_score = -math.inf
    depth = 1
    completed_depth = 0

    # Iterative deepening: increase search depth until time runs out
    try:
        while depth <= max_depth:
            if time.monotonic() > deadline:
                break
            
            # Search at current depth
            local_best_col, local_best_score = _search_depth(
                board, player, depth, valid_cols, state, deadline
            )
            
            # Only update if search completed successfully
            best_score = local_best_score
            best_col = local_best_col
            completed_depth = depth
            depth += 1
            
    except TimeoutError:
        pass  # Return the best move found so far if time runs out
    
    # Store the best move for the root position with the completed depth
    if completed_depth > 0:
        state.store_position(board, completed_depth, best_score, PlayerAction(best_col), 'exact')
    
    return PlayerAction(best_col), state

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
    opponent_count = np.count_nonzero(window == _other(player))
    empty_count = np.count_nonzero(window == NO_PLAYER)
    
    # Terminal states
    if player_count == 4:
        return 1000.0  # Win - much higher value
    if opponent_count == 4:
        return -1000.0  # Opponent win
    
    # Strong threats
    if player_count == 3 and empty_count == 1:
        return 50.0   # Strong threat - increased from 5.0
    if opponent_count == 3 and empty_count == 1:
        return -45.0  # Must block - increased penalty
    
    # Medium threats  
    if player_count == 2 and empty_count == 2:
        # Check if pieces are connected (more valuable)
        if _are_pieces_connected(window, player):
            return 10.0  # Connected pair
        else:
            return 6.0   # Separated pair
    if opponent_count == 2 and empty_count == 2:
        if _are_pieces_connected(window, _other(player)):
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
        elif board[row, center_col] == _other(player):
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
            elif board[row, col] == _other(player):
                height_bonus = (BOARD_ROWS - row) * 0.5
                score -= (col_weight + height_bonus) * 0.8
    
    return score


def _detect_multiple_threats(board: np.ndarray, player: BoardPiece) -> float:
    """
    Detect and score positions that create multiple winning threats.
    
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


def _other(player: BoardPiece) -> BoardPiece:
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

def _heuristic(board: np.ndarray, player: BoardPiece) -> float:
    """
    Enhanced heuristic evaluation of the board for the given player.
    
    Considers:
    - Pattern-based scoring (windows)
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
    
    # 3. Positional weights (new)
    score += _score_positional_weights(board, player)
    
    # 4. Multiple threat detection (new)
    score += _detect_multiple_threats(board, player)
    score -= _detect_multiple_threats(board, _other(player))  # Penalty for opponent threats
    
    # 5. Connectivity bonus (new)
    score += _score_connectivity(board, player)
    score -= _score_connectivity(board, _other(player)) * 0.8  # Penalty for opponent connectivity
    
    return score
