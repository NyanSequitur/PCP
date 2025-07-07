### Core Modules

1. **`agent.py`** - Main entry point and high-level coordination
   - `generate_move_time_limited()` - Primary function for move generation
   - `generate_move()` - Convenience function with default parameters
   - `create_minimax_agent()` - Factory function for creating configured agents

2. **`search.py`** - Core search algorithms
   - `negamax_search()` - Main negamax implementation with alpha-beta pruning
   - `search_at_depth()` - Search at a specific depth
   - `get_transposition_flag()` - Utility for transposition table bounds

3. **`search_continuation.py`** - Iterative deepening and search management
   - `iterative_deepening_search()` - Main iterative deepening loop
   - `should_extend_search()` - Selective search extension logic
   - `get_search_statistics()` - Performance metrics

4. **`heuristics.py`** - Position evaluation functions
   - `evaluate_board()` - Main board evaluation function
   - Window scoring functions (`_score_window`, `_score_horizontal_windows`, etc.)
   - Advanced heuristics (threat detection, connectivity, positional weights)

5. **`transposition_table.py`** - Position caching and symmetry handling
   - `TranspositionTable` class for storing computed positions
   - `TranspositionEntry` dataclass for table entries
   - Symmetry detection and canonical board representation

6. **`move_ordering.py`** - Move ordering for better search efficiency
   - `get_valid_columns()` - Find valid moves
   - `order_moves()` - Order moves for better alpha-beta pruning
   - `MoveOrderingCache` - Cache for move orderings

7. **`saved_state.py`** - Persistent state management
   - `MinimaxSavedState` class combining all persistent components
   - State management and cleanup utilities

## Key Features

### Negamax with Alpha-Beta Pruning
The search uses the negamax algorithm, which is a simplified version of minimax that takes advantage of the zero-sum property of Connect Four. Alpha-beta pruning dramatically reduces the search space.

### Iterative Deepening
The agent uses iterative deepening to gradually increase search depth until time runs out. This provides:
- Quick initial moves from shallow search
- Ability to interrupt search at any time
- Better move ordering from previous iterations

### Transposition Table
Positions are cached in a transposition table to avoid recomputing the same positions. The table includes:
- Symmetry detection (horizontal mirroring)
- Canonical board representation
- Automatic cleanup when table gets too large

### Advanced Heuristics
The evaluation function considers multiple factors:
- Traditional window-based scoring (4-cell patterns)
- Positional weights (center columns preferred)
- Multiple threat detection
- Piece connectivity
- Center column control

### Move Ordering
Moves are ordered to improve alpha-beta pruning efficiency:
1. Best move from transposition table
2. Center columns (generally stronger in Connect Four)
3. Remaining columns by distance from center

## Usage

### Basic Usage
```python
from agents.agent_minimax import generate_move_time_limited

# Generate a move with 5-second time limit
move, state = generate_move_time_limited(board, player, time_limit_secs=5.0)
```

### Advanced Usage
```python
from agents.agent_minimax import create_minimax_agent

# Create a configured agent
move_function, initial_state = create_minimax_agent(
    time_limit=10.0,
    max_depth=15,
    max_table_size=2000000
)

# Use the agent
move, updated_state = move_function(board, player, initial_state)
```

### Using Individual Components
```python
from agents.agent_minimax.heuristics import evaluate_board
from agents.agent_minimax.transposition_table import TranspositionTable
from agents.agent_minimax.search import negamax_search

# Evaluate a position
score = evaluate_board(board, player)

# Use transposition table directly
tt = TranspositionTable()
tt.store_position(board, depth=5, value=10.0, best_move=PlayerAction(3), flag='exact')
```

## Testing

The modular structure makes it easier to test individual components:

```bash
# Run all tests
python -m pytest tests/

# Test specific modules
python -m pytest tests/test_heuristics.py
python -m pytest tests/test_transposition_table.py
```

## Performance Considerations

- **Transposition Table**: Automatically manages memory usage
- **Move Ordering**: Significantly improves search efficiency
- **Iterative Deepening**: Provides good anytime behavior
- **Symmetry Detection**: Reduces memory usage by ~50%

## Future Enhancements

The modular structure makes it easy to add new features:
- Opening book integration
- Endgame tablebase
- Advanced search extensions
- Machine learning-based evaluation
- Parallel search
- Better time management

## File Structure

```
agents/agent_minimax/
├── __init__.py              # Module exports
├── agent.py                 # Main agent interface
├── search.py                # Core search algorithms
├── search_continuation.py   # Iterative deepening
├── heuristics.py           # Position evaluation
├── transposition_table.py  # Position caching
├── move_ordering.py        # Move ordering utilities
├── saved_state.py          # State management
├── minimax.py              # Legacy compatibility
└── README.md               # This file
```

This modular architecture makes the code more maintainable, testable, and understandable while preserving all the sophisticated features of the original implementation.