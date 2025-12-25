# Policy Large - Supervised Learning from MCTS

Train a policy network to match MCTS visit distributions from collected game logs.

## Usage

### Training

Train a model on MCTS collected games:

```bash
policy-large-train --mcts-logs-dir scoundrel/rl/mcts/logs/collected_games --epochs 100
```

Options:
- `--mcts-logs-dir`: Directory containing MCTS log JSON files (default: `scoundrel/rl/mcts/logs/collected_games`)
- `--max-games`: Maximum number of games to load (default: all)
- `--batch-size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints/`)
- `--resume-from`: Path to checkpoint to resume from
- `--no-tensorboard`: Disable TensorBoard logging

### Viewing

Run interactive viewer with a trained checkpoint:

```bash
policy-large-viewer --checkpoint checkpoints/policy_large_epoch_100.pt
```

Options:
- `--checkpoint`: Path to checkpoint file (required)
- `--label`: Label shown in UI (default: "Policy Large")
- `--seed`: Seed for deterministic deck shuffling

### TensorBoard

View training metrics:

```bash
tensorboard --logdir scoundrel/rl/alpha_scoundrel/policy/policy_large/runs
```

## Architecture

- Transformer encoder for dungeon stack sequence
- MLP for current room & status features
- Single actor head (no critic) outputting action logits
- Trained with cross-entropy loss on MCTS visit distributions

## Data Format

Requires MCTS log files in JSON format with:
- `game_state`: Serialized game state
- `mcts_stats`: List of action visit counts
- `selected_action`: Action taken

See `REQUIREMENTS.md` for detailed data format specification.

