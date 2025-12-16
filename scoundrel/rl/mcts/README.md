# MCTS Implementation for Scoundrel

Monte Carlo Tree Search (MCTS) agent for playing the Scoundrel card game.

## Overview

This implementation uses classic MCTS with four phases:
1. **Selection**: Traverse the tree using UCB1 to find a promising node
2. **Expansion**: Add a new child node for an unexplored action
3. **Simulation**: Play out the game from the new node using a rollout policy
4. **Backpropagation**: Update all nodes in the path with the simulation result

## Directory Structure

```
mcts/
├── __init__.py           # Package initialization
├── constants.py          # Configuration constants
├── mcts_node.py          # MCTS tree node implementation
├── mcts_agent.py         # MCTS algorithm implementation
├── mcts.py               # Main training/evaluation script
├── viewer.py             # Interactive viewer for watching agent play
├── scripts/
│   └── train.sh          # Training shell script
├── logs/                 # Statistics and logs (created on first run)
└── README.md             # This file
```

## Quick Start

### Run the Interactive Viewer

Watch the MCTS agent play Scoundrel step-by-step:

```bash
# From the project root
python -m scoundrel.rl.mcts.viewer

# With custom number of simulations
python -m scoundrel.rl.mcts.viewer --num-simulations 200
```

Controls:
- `Space` or `Enter`: Execute next action
- `r`: Restart game
- `q`: Quit

### Run Batch Evaluation

Evaluate the MCTS agent over multiple games:

```bash
# Using the shell script
./scoundrel/rl/mcts/scripts/train.sh

# Or directly with Python
python -m scoundrel.rl.mcts.mcts

# With custom parameters
python -m scoundrel.rl.mcts.mcts --num-simulations 200 --num-games 100
```

### Clear Logs

```bash
./scoundrel/rl/mcts/scripts/train.sh --clear
```

## Configuration

Edit `constants.py` to adjust MCTS parameters:

- `MCTS_NUM_SIMULATIONS`: Number of simulations per move (default: 100)
- `MCTS_EXPLORATION_CONSTANT`: UCB1 exploration parameter (default: 1.414)
- `MCTS_MAX_DEPTH`: Maximum simulation depth (default: 200)
- `USE_RANDOM_ROLLOUT`: Use random vs heuristic rollout policy (default: True)

## Command-Line Options

### Viewer Options

```bash
python -m scoundrel.rl.mcts.viewer --help
```

- `--num-simulations`: Number of MCTS simulations per move

### Training Options

```bash
python -m scoundrel.rl.mcts.mcts --help
```

- `--num-simulations`: Number of MCTS simulations per move
- `--num-games`: Number of games to play
- `--save-interval`: Save statistics every N games
- `--resume-from`: Path to statistics file to resume from
- `--quiet`: Disable verbose output

## Statistics

Statistics are saved to `logs/mcts_stats.json` and include:
- Games played
- Average score
- Best/worst scores
- Win rate
- Full score history

## Architecture

### Shared Components

The MCTS implementation shares components with other RL methods:
- `scoundrel.rl.translator`: State encoding and action decoding
- `scoundrel.rl.viewer`: Interactive viewer utilities

### MCTS-Specific Components

- **MCTSNode**: Tree node with UCB1 child selection
- **MCTSAgent**: Core MCTS algorithm with configurable policies
- **MCTSStatistics**: Performance tracking and analysis

## Comparison with Transformer PPO

Unlike the transformer_mlp implementation which learns a neural network policy through training, MCTS is a search-based method that:
- Requires no training phase
- Plans ahead by building a search tree
- Can be run immediately with any computational budget
- Performance scales with number of simulations per move

## Performance

Typical performance with 100 simulations per move:
- Average score: Varies based on game complexity
- Time per move: ~1-3 seconds
- Win rate: Tracked in statistics

Increase `MCTS_NUM_SIMULATIONS` for better performance at the cost of computation time.



