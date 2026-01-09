# Self-Play Training - REINFORCE Policy Gradients

Iterative self-play training with REINFORCE policy gradients for AlphaGo-style agents.

## Overview

This module implements **iterative self-play training** using **REINFORCE policy gradients**:

1. **Generate self-play games** using current best AlphaGo agent
2. **Train policy network** with REINFORCE: Win (+1) reinforces actions, Loss (-1) discourages actions
3. **Train value network** on game outcomes
4. **Evaluate** and update best checkpoints
5. **Repeat** with improved networks

## Usage

### Training

Start self-play training with default settings:

```bash
self-play --verbose
```

Run for specific number of iterations:

```bash
self-play --max-iterations 50 --verbose
```

Resume from a specific iteration:

```bash
self-play --resume-from 5 --verbose
```

### Options

#### Training Configuration
- `--max-iterations`: Maximum iterations to run (default: 1000)
- `--checkpoint-dir`: Base directory for checkpoints (default: `scoundrel/rl/alpha_scoundrel/self_play/checkpoints`)
- `--resume-from`: Resume from specific iteration (default: start from 1)
- `--num-parallel-games`: Number of games to generate simultaneously (default: 8)

#### Model Configuration
- `--policy-checkpoint`: Initial policy checkpoint path (default: built-in)
- `--value-checkpoint`: Initial value checkpoint path (default: built-in)
- `--policy-small-checkpoint`: Policy small checkpoint for fast rollouts (default: built-in)

#### Monitoring & Logging
- `--verbose`, `-v`: Print detailed progress with colored concurrent game status
- `--no-tensorboard`: Disable TensorBoard logging

### TensorBoard Monitoring

View training metrics in real-time:

```bash
tensorboard --logdir scoundrel/rl/alpha_scoundrel/self_play/runs
```

## Training Loop Details

### Step 1: Self-Play Game Generation
- Uses current best AlphaGo agent to play against itself
- Generates games with exploration noise (Dirichlet + epsilon)
- **Parallel generation**: Up to 8 games simultaneously (configurable)
- Each game runs in separate process to avoid GPU/MPS conflicts

### Step 2: Policy Training (REINFORCE)
- **Objective**: Maximize expected reward using policy gradients
- **Reward**: Win = +1, Loss = -1 (normalized or scaled)
- **Algorithm**: REINFORCE with optional entropy bonus
- **Loss**: Policy gradient loss with entropy regularization
- **Optimization**: Adam with gradient clipping

### Step 3: Value Training
- **Objective**: Predict game outcomes from board positions
- **Architecture**: Large value network (transformer + MLP)
- **Loss**: MSE between predicted and actual game outcomes
- **Optimization**: Adam with cosine annealing LR schedule

### Step 4: Evaluation
- Evaluate new agent against fixed opponent
- Update best checkpoints if improved
- Continue training with improved networks

## Architecture

### Policy Network (REINFORCE Training)
- **Input**: Game state (dungeon stack, current room, status)
- **Architecture**: Large policy network (transformer + MLP)
- **Output**: Action probabilities (actor head only)
- **Training**: REINFORCE policy gradients with entropy bonus

### Value Network
- **Input**: Game state (same as policy)
- **Architecture**: Large value network (transformer + MLP)
- **Output**: Predicted game outcome (-1 to +1)
- **Training**: Supervised regression on game results

### AlphaGo Agent
- **Policy**: Large policy network for high-quality moves
- **Policy Small**: Fast policy for MCTS rollouts
- **Value**: Value network for position evaluation
- **MCTS**: Monte Carlo Tree Search with neural guidance
- **Exploration**: Dirichlet noise during self-play

## Data Format

Self-play generates training data in JSON format:

```json
{
  "metadata": {
    "seed": 12345,
    "num_simulations": 80,
    "final_score": 15,
    "num_turns": 25,
    "final_state": {...},
    "agent_type": "alphago_self_play"
  },
  "events": [
    {
      "turn": 0,
      "game_state": {...},
      "mcts_stats": [...],
      "selected_action": 5,
      "cache_stats": {...}
    }
  ]
}
```

## Configuration

Key parameters in `constants.py`:

- `SELF_PLAY_GAMES_PER_ITERATION = 10` - Games per iteration
- `SELF_PLAY_SIMULATIONS = 80` - MCTS simulations per move
- `SELF_PLAY_PARALLEL_GAMES = 8` - Parallel game generation
- `POLICY_EPOCHS_PER_ITERATION = 20` - Policy training epochs
- `VALUE_EPOCHS_PER_ITERATION = 20` - Value training epochs
- `BATCH_SIZE = 256` - Training batch size

## Requirements

- PyTorch with CUDA/MPS support (GPU recommended)
- Sufficient RAM for parallel game generation
- Disk space for game logs and checkpoints
- TensorBoard for monitoring (optional)

## Output

Training creates organized directory structure:
```
checkpoints/
├── iteration_001/
│   ├── policy_large_best.pt
│   └── value_large_best.pt
├── iteration_002/
│   └── ...
├── best/                    # Current best models
│   ├── policy_large_best.pt
│   └── value_large_best.pt
└── results.txt             # Training summary

runs/                       # TensorBoard logs
├── self_play_20240101_120000/
│   ├── iter_001/
│   ├── iter_002/
│   └── ...
└── ...

iteration_001/games/        # Self-play game logs
├── 12345.json
├── 12346.json
└── ...
```

Each iteration includes comprehensive results in `results.txt` with win rates, training metrics, and evaluation scores.
