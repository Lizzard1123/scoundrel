# Policy Small - Advanced Supervised Learning from MCTS

Train a policy network with modern training techniques to match MCTS visit distributions from collected game logs.

## Usage

### Training

Train a model on MCTS collected games with advanced features:

```bash
policy-small-train --mcts-logs-dir scoundrel/rl/mcts/logs/collected_games --epochs 500
```

Options:
- `--mcts-logs-dir`: Directory containing MCTS log JSON files (default: `scoundrel/rl/mcts/logs/collected_games`)
- `--max-games`: Maximum number of games to load (default: all)
- `--batch-size`: Batch size (default: 256)
- `--epochs`: Number of epochs (default: 500)
- `--lr`: Learning rate (default: 1e-3)
- `--temperature`: Temperature for sharpening targets (default: 0.5, < 1.0 sharpens)
- `--use-q-weights`: Weight visits by Q-values from MCTS
- `--hard-loss-weight`: Weight for hard classification loss (default: 1.0)
- `--focal-gamma`: Gamma parameter for focal MSE loss (default: 2.0)
- `--dropout-rate`: Dropout rate for regularization (default: 0.1)
- `--max-grad-norm`: Maximum gradient norm for clipping (default: 1.0)
- `--warmup-epochs`: Number of warmup epochs (default: 0)
- `--min-lr-ratio`: Minimum LR ratio for decay (default: 1.0)
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints/`)
- `--resume-from`: Path to checkpoint to resume from
- `--no-tensorboard`: Disable TensorBoard logging

### Viewing

Run interactive viewer with a trained checkpoint:

```bash
policy-small-viewer --checkpoint checkpoints/policy_small_epoch_500.pt
```

Options:
- `--checkpoint`: Path to checkpoint file (required)
- `--label`: Label shown in UI (default: "Policy Small")
- `--seed`: Seed for deterministic deck shuffling

### TensorBoard

View training metrics:

```bash
tensorboard --logdir scoundrel/rl/alpha_scoundrel/policy/policy_small/runs
```

## Architecture

- Single FC layer with dropout regularization
- Advanced training with gradient clipping and modern loss functions
- Hybrid loss: Focal MSE + hard classification on best actions
- Learning rate scheduling with warmup and cosine decay
- Temperature sharpening for more decisive training targets

## Training Features

### Advanced Loss Functions
- **Focal MSE**: Adapts focal loss to regression, focuses on hard examples with `|y-ŷ|^γ * (y-ŷ)²`
- **Hybrid Loss**: Combines soft distribution matching with hard best-action classification
- **Temperature Sharpening**: Makes training targets more decisive (< 1.0 sharpens toward one-hot)

### Regularization
- **Dropout**: Applied before final layer for robustness
- **Gradient Clipping**: Prevents exploding gradients with configurable max norm
- **Learning Rate Scheduling**: Warmup + cosine decay for stable convergence

### Data Augmentation
- **Target Sharpening**: Temperature parameter controls target distribution sharpness
- **Q-Value Weighting**: Optional weighting of MCTS visits by their Q-values

## Data Format

Requires MCTS log files in JSON format with:
- `game_state`: Serialized game state
- `mcts_stats`: List of action visit counts with optional Q-values
- `selected_action`: Action taken

See `REQUIREMENTS.md` for detailed data format specification.

