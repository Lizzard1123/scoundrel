# Value Large - Supervised Learning from MCTS Final Scores

Train a value network to predict expected game outcome (final score) given a game state.

## Usage

### Training

Train a model on MCTS collected games:

```bash
value-large-train --mcts-logs-dir scoundrel/rl/mcts/logs/collected_games --epochs 100
```

Options:
- `--mcts-logs-dir`: Directory containing MCTS log JSON files (default: `scoundrel/rl/mcts/logs/collected_games`)
- `--max-games`: Maximum number of games to load (default: all)
- `--batch-size`: Batch size (default: 128)
- `--epochs`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 5e-4)
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints/`)
- `--resume-from`: Path to checkpoint to resume from
- `--no-tensorboard`: Disable TensorBoard logging

### Viewing

Run interactive viewer with a trained checkpoint:

```bash
value-large-viewer --checkpoint checkpoints/value_large_epoch_100.pt
```

Options:
- `--checkpoint`: Path to checkpoint file (required)
- `--label`: Label shown in UI (default: "Value Large")
- `--seed`: Seed for deterministic deck shuffling

### TensorBoard

View training metrics:

```bash
tensorboard --logdir scoundrel/rl/alpha_scoundrel/value/value_large/runs
```

## Architecture

- Transformer encoder for dungeon stack sequence
- MLP for current room & status features
- Single value head outputting scalar expected value
- Trained with MSE loss on final game scores

## Data Format

Requires MCTS log files in JSON format with:
- `game_state`: Serialized game state
- `metadata.final_score`: Final score of the game (used as label for all states in that game)

Each state in a game is labeled with the final score of that game, teaching the network to predict expected outcome from any given state.

## Troubleshooting

### MPS Device Warnings

If you're running on Apple Silicon (MPS) and see warnings about `aten::_nested_tensor_from_mask_left_aligned` not being implemented, this is a known PyTorch limitation. The warning occurs when TensorBoard tries to log the model graph but doesn't affect training.

To suppress the warning, you can set the environment variable before running:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
value-large-train --mcts-logs-dir scoundrel/rl/mcts/logs/collected_games --epochs 100
```

Alternatively, disable TensorBoard graph logging by using the `--no-tensorboard` flag if you don't need TensorBoard visualization.

Note: Training will continue normally despite the warning - it only affects the model graph visualization in TensorBoard.

