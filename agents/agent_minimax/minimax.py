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
    
    def __init__(self):
        self.transposition_table: Dict[str, TranspositionEntry] = {}
        self.move_ordering: Dict[str, list[PlayerAction]] = {}
    
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
        canonical = self._get_canonical_board(board)
        return not np.array_equal(canonical, board)
    
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
    
    def lookup_position(self, board: np.ndarray, depth: int, alpha: float, beta: float) -> Tuple[bool, float]:
        """Look up a position in the transposition table, handling symmetry."""
        board_hash = self.get_board_hash(board)
        if board_hash not in self.transposition_table:
            return False, 0.0
        
        entry = self.transposition_table[board_hash]
        if entry.depth < depth:
            return False, 0.0
        
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
        for direction in [-1, 1]:
            col = center + direction * offset
            if 0 <= col < BOARD_COLS and col in remaining_cols:
                ordered_moves.append(col)
                remaining_cols.remove(col)
                break
    
    # Add any remaining columns
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
    time_limit_secs: float = 5.0
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

    Returns
    -------
    Tuple[PlayerAction, Optional[SavedState]]
        The chosen move and updated state.
    """
    start_time = time.monotonic()
    deadline = start_time + time_limit_secs
    
    # Initialize search
    state, valid_cols, best_col = _initialize_search(board, saved_state)
    best_score = -math.inf
    depth = 1

    # Iterative deepening: increase search depth until time runs out
    try:
        while True:
            if time.monotonic() > deadline:
                break
            
            # Search at current depth
            local_best_col, local_best_score = _search_depth(
                board, player, depth, valid_cols, state, deadline
            )
            
            best_score = local_best_score
            best_col = local_best_col
            depth += 1
            
    except TimeoutError:
        pass  # Return the best move found so far if time runs out
    
    # Store the best move for the root position
    state.store_position(board, depth - 1, best_score, PlayerAction(best_col), 'exact')
    
    return PlayerAction(best_col), state

def _score_window(window: np.ndarray, player: BoardPiece) -> float:
    """
    Score a 4-cell window for the given player.
    
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
    
    if player_count == 4:
        return 100.0  # Win
    if player_count == 3 and empty_count == 1:
        return 5.0   # Three in a row
    if player_count == 2 and empty_count == 2:
        return 2.0   # Two in a row
    if opponent_count == 3 and empty_count == 1:
        return -4.0  # Block opponent
    return 0.0


def _score_center_column(board: np.ndarray, player: BoardPiece) -> float:
    """
    Score the center column control for the given player.
    
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
    center_count = int(np.count_nonzero(board[:, center_col] == player))
    return center_count * 3.0


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
    Heuristic evaluation of the board for the given player.
    Considers center column control and open-ended 2/3-in-a-row windows.

    Parameters
    ----------
    board : np.ndarray
        The game board.
    player : BoardPiece
        The player to evaluate for.

    Returns
    -------
    float
        Heuristic score.
    """
    score = 0.0
    
    # Score center column control
    score += _score_center_column(board, player)
    
    # Score all window types
    score += _score_horizontal_windows(board, player)
    score += _score_vertical_windows(board, player)
    score += _score_diagonal_windows(board, player)
    
    return score
