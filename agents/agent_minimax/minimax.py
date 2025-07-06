"""
Minimax agent for Connect Four with iterative deepening and alpha-beta pruning.

Implements a time-limited move generator using Negamax with alpha-beta pruning and a simple heuristic.
"""
import io
import zipfile
import urllib.request
import warnings
from functools import lru_cache

import time
import math
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

# ── LZW decompressor: accept either package name ──────────────────────
try:
    from unlzw import unlzw                # PyPI (‘unlzw’)
except ModuleNotFoundError:                # pragma: no cover
    try:
        from unlzw3 import unlzw           # conda-forge (‘unlzw3’)
    except ModuleNotFoundError:
        unlzw = None                       # graceful fallback

from game_utils import (
    BOARD_COLS, BOARD_ROWS, NO_PLAYER, PLAYER1, PLAYER2,
    apply_player_action, check_end_state, GameState,
    check_move_status, MoveStatus, PlayerAction, BoardPiece, SavedState
)



# Remark: Awesome! There really isn't much for me to complain about, great work!
# Take the remarks I left as suggestions, maybe you find them helpful.



# ----------------------------------------------------------------------
# Canonical-board helpers (module-level to avoid recursion in loader)
# ----------------------------------------------------------------------
def _canonical_board(board: np.ndarray) -> np.ndarray:
    """
    Return the lexicographically smaller of *board* and its horizontal
    mirror.  This is the canonical form used for hashing / symmetry.
    """
    mirrored = np.fliplr(board)
    return board if tuple(board.flatten()) <= tuple(mirrored.flatten()) else mirrored


def _board_hash(board: np.ndarray) -> str:
    """
    Hex string suitable as a key in transposition table or opening book.
    """
    return _canonical_board(board).tobytes().hex()


@lru_cache(maxsize=1)
def _load_opening_book() -> dict[str, int]:
    """
    Download & parse John Tromp’s 8-ply Connect-4 database (once).

    Returns
    -------
    dict[str,int]
        canonical board-hash → game-theoretic value for Player 1
        (+1 win · 0 draw · −1 loss).

    If anything fails (network, zip, LZW, missing library) a single
    warning is issued and an *empty* dict is returned so the agent
    silently falls back to heuristic play.
    """
    print("Starting to load opening book...")  # Debug print

    # ——— ensure we have an LZW decompressor ——————————————
    if unlzw is None:
        warnings.warn(
            "Neither 'unlzw' nor 'unlzw3' is installed – opening book "
            "disabled, falling back to heuristic search.",
            RuntimeWarning, stacklevel=2
        )
        print("LZW decompressor not available, returning empty book.")  # Debug print
        return {}

    url = "https://archive.ics.uci.edu/static/public/26/connect+4.zip"
    try:
        print(f"Downloading opening book from {url}...")  # Debug print
        with urllib.request.urlopen(url, timeout=10) as resp:
            zip_data = resp.read()
        print("Download successful, extracting ZIP file...")  # Debug print
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            comp = zf.read("connect-4.data.Z")
        print("ZIP extraction successful, decompressing data...")  # Debug print
        raw = unlzw(comp)                           # bytes
    except Exception as exc:                        # noqa: BLE001
        warnings.warn(
            f"Opening-book download failed ({exc}) – "
            "falling back to heuristic search.",
            RuntimeWarning, stacklevel=2
        )
        print(f"Failed to download or process opening book: {exc}")  # Debug print
        return {}

    # ——— parse file (67 557 lines) ——————————————————————
    print("Parsing opening book data...")  # Debug print
    book: dict[str, int] = {}
    for line in raw.decode("ascii").strip().splitlines():
        parts = line.split(",")
        if len(parts) != 43:                        # 42 squares + outcome
            print(f"Skipping malformed line: {line}")  # Debug print
            continue                                # defensive

        cells, outcome = parts[:-1], parts[-1]

        board = np.full((BOARD_ROWS, BOARD_COLS),
                        NO_PLAYER, dtype=np.int8)
        idx = 0
        for col in range(BOARD_COLS):               # a..g
            for row in range(BOARD_ROWS):           # 1..6 (bottom→top)
                token = cells[idx]
                if token == "x":
                    board[row, col] = PLAYER1
                elif token == "o":
                    board[row, col] = PLAYER2
                idx += 1

        value = {"win": 1, "draw": 0, "loss": -1}[outcome]
        book[_board_hash(board)] = value

    print(f"Opening book loaded successfully with {len(book)} entries.")  # Debug print
    return book



































@dataclass
class TranspositionEntry:
    """Entry in the transposition table."""
    value: float
    depth: int
    best_move: Optional[PlayerAction]
    flag: str                     # 'exact', 'lower', 'upper'


class MinimaxSavedState(SavedState):
    """
    Saved state for the minimax agent:
    – transposition tables
    – optional 8-ply opening book
    """

    def __init__(self):
        # ── transposition-table bits ───────────────────────────────────
        self.transposition_table: Dict[str, TranspositionEntry] = {}
        self.move_ordering: Dict[str, list[PlayerAction]] = {}

        # ── opening-book bits ──────────────────────────────────────────
        self.opening_book: dict[str, int] = _load_opening_book()
        self.use_opening_book: bool = bool(self.opening_book)

    # ───────────── symmetry / hashing helpers (wrapper) ────────────────
    def get_board_hash(self, board: np.ndarray) -> str:
        return _board_hash(board)

    def _is_board_mirrored(self, board: np.ndarray) -> bool:
        return not np.array_equal(_canonical_board(board), board)

    def _mirror_move(self, move: PlayerAction) -> PlayerAction:
        return PlayerAction(BOARD_COLS - 1 - int(move))

    # ─────────────── transposition-table API ───────────────────────────
    def store_position(self, board: np.ndarray, depth: int,
                       value: float, best_move: Optional[PlayerAction],
                       flag: str):
        board_hash = _board_hash(board)
        if best_move is not None and self._is_board_mirrored(board):
            best_move = self._mirror_move(best_move)

        self.transposition_table[board_hash] = TranspositionEntry(
            value=value, depth=depth, best_move=best_move, flag=flag
        )

    def lookup_position(self, board: np.ndarray, depth: int,
                        alpha: float, beta: float) -> Tuple[bool, float]:
        entry = self.transposition_table.get(_board_hash(board))
        if entry is None or entry.depth < depth:
            return False, 0.0

        if entry.flag == 'exact':
            return True, entry.value
        if entry.flag == 'lower' and entry.value >= beta:
            return True, entry.value
        if entry.flag == 'upper' and entry.value <= alpha:
            return True, entry.value
        return False, 0.0

    def get_best_move(self, board: np.ndarray) -> Optional[PlayerAction]:
        entry = self.transposition_table.get(_board_hash(board))
        if entry and entry.best_move is not None:
            return (self._mirror_move(entry.best_move)
                    if self._is_board_mirrored(board) else entry.best_move)
        return None

    def get_move_ordering(self, board: np.ndarray) -> list[PlayerAction]:
        return self.move_ordering.get(_board_hash(board), [])

    def store_move_ordering(self, board: np.ndarray,
                            moves: list[PlayerAction]):
        self.move_ordering[_board_hash(board)] = moves

    # ─────────────── opening-book convenience ──────────────────────────
    def opening_value(self, board: np.ndarray) -> Optional[int]:
        if not self.use_opening_book:
            return None
        return self.opening_book.get(_board_hash(board))



















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
    Negamax with alpha-beta pruning and transposition table.
    """
    if time.monotonic() > deadline:
        raise TimeoutError

    # ─── TT probe ──────────────────────────────────────────────────────
    found, tt_val = state.lookup_position(board, depth, alpha, beta)
    if found:
        return tt_val
    orig_alpha = alpha

    # ─── terminal checks ───────────────────────────────────────────────
    prev_player = PLAYER1 if turn == PLAYER2 else PLAYER2
    end_state = check_end_state(board, prev_player)
    if end_state == GameState.IS_WIN:
        return -math.inf
    if end_state == GameState.IS_DRAW:
        return 0.0
    if depth == 0:
        return _heuristic(board, turn, state)

    # ─── recursive search ──────────────────────────────────────────────
    best_val = -math.inf
    best_move = None
    valid_cols = _get_valid_columns(board)
    for col in _order_moves(board, valid_cols, state):
        if time.monotonic() > deadline:
            raise TimeoutError

        child = board.copy()
        apply_player_action(child, PlayerAction(col), turn)
        score = -_negamax(child, depth - 1, -beta, -alpha,
                          PLAYER1 if turn == PLAYER2 else PLAYER2,
                          state, deadline)

        if score > best_val:
            best_val, best_move = score, PlayerAction(col)
        alpha = max(alpha, score)
        if alpha >= beta:
            break                       # β-cutoff

    # ─── store to TT ───────────────────────────────────────────────────
    flag = ('upper' if best_val <= orig_alpha else
            'lower' if best_val >= beta else
            'exact')
    state.store_position(board, depth, best_val, best_move, flag)
    return best_val


















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
    time_limit_secs: float = 10.0
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
    max_depth_reached = 0
    tt_hits = 0
    book_hits = 0
    
    # Track statistics
    original_tt_size = len(state.transposition_table)

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
            max_depth_reached = depth
            depth += 1
            
    except TimeoutError:
        pass  # Return the best move found so far if time runs out
    
    # Calculate statistics
    new_tt_entries = len(state.transposition_table) - original_tt_size
    used_opening_book = state.use_opening_book and state.opening_value(board) is not None
    
    # Print one clean summary
    print(f"Minimax AI: Move {best_col} | Depth {max_depth_reached} | "
          f"Book: {'Yes' if used_opening_book else 'No'} | "
          f"TT: {len(state.transposition_table)} entries (+{new_tt_entries})")
    
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

def _heuristic(board: np.ndarray, player: BoardPiece,
               state: Optional[MinimaxSavedState] = None) -> float:
    """
    Evaluate *board* for *player*.

    • If the position is present in the 8-ply opening book, return a huge
      fixed score (scaled to *player*’s perspective) so the search
      immediately prefers it.  The value is also cached in the TT.

    • Otherwise fall back to the original pattern-based heuristic.
    """
    if state is not None:
        book_val = state.opening_value(board)
        if book_val is not None:
            sign = 1 if player == PLAYER1 else -1
            score = sign * book_val * 1_000_000.0   # dominates any eval
            # cache exact perfect info
            state.store_position(board, 0, score, None, 'exact')
            return score

    # ─── traditional heuristic ────────────────────────────────────────
    score = 0.0
    score += _score_center_column(board, player)
    score += _score_horizontal_windows(board, player)
    score += _score_vertical_windows(board, player)
    score += _score_diagonal_windows(board, player)
    return score
