# connect-four-engine

A complete Connect Four implementation with multiple AI agents, extensive test coverage, and performance-oriented search optimizations.

## Purpose and Functionality

The project provides a ready-to-use game engine for Connect Four. It includes:

* Rules enforcement, move validation, and win detection
* Three player types: minimax AI (alpha–beta), random AI, and human
* Optimizations such as transposition tables with symmetry reduction, iterative deepening, move ordering, and search continuation
* 181 unit tests to ensure functional correctness
* Benchmark scripts for win-rate and speed analysis

## Installation

### Prerequisites

* Python 3.8 or newer
* NumPy
* pytest (for running tests)

### Steps

```bash
git clone https://github.com/NyanSequitur/PCP.git
cd PCP
pip install numpy pytest
```

## Usage

### Interactive Game

```bash
python main.py          # select players in the terminal
```

### Quick Human vs AI

```bash
python minimal_main.py
```

### Programmatic Example

```python
from game_utils import initialize_game_state, apply_player_action, PLAYER1
from agents.agent_minimax import generate_move_time_limited

board = initialize_game_state()
move, state = generate_move_time_limited(board, PLAYER1, time_limit_secs=5)
apply_player_action(board, move, PLAYER1)
```

### Configurable Minimax Agent

```python
from agents.agent_minimax import create_minimax_agent
move_fn, agent_state = create_minimax_agent(time_limit=10.0,
                                            max_depth=15,
                                            max_table_size=2_000_000)
move, agent_state = move_fn(board, player, agent_state)
```

## Testing

```bash
pytest                     # run the full suite
pytest tests/test_game_utils.py
```

## Benchmarking

```bash
python benchmark_symmetry.py   # minimax vs random win rate
python benchmark_win_rate.py   # comprehensive win-rate tests
python benchmark_scales.py     # search depth and memory scaling
```

## Architecture

```
game_utils.py                core engine and helpers
main.py                      interactive CLI
agents/                      AI implementations
│   ├── agent_minimax/       minimax agent and submodules
│   ├── agent_random/        random player
│   └── agent_human_user/    human input handler
tests/                       unit and integration tests
benchmarks/                  performance scripts
```