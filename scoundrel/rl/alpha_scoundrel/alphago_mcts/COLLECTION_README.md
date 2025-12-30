# AlphaGo MCTS Collection and Evaluation

This directory contains scripts for collecting and evaluating game data using the AlphaGo-style MCTS agent.

## Collection Script

The `collect.py` script runs AlphaGo MCTS on multiple games and saves comprehensive game data including:
- Game state snapshots at each turn
- MCTS statistics (visits, values, policy priors)
- Cache statistics (hit rates)
- Final scores and game outcomes

### Usage

```bash
# Collect 10 games with default settings
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.collect --num-games 10 --verbose

# Collect games with custom simulations
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.collect --num-games 100 --num-simulations 1000 --verbose

# Collect games to a specific directory
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.collect --num-games 50 --output-dir ./my_games --verbose

# Collect with specific seed for reproducibility
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.collect --num-games 10 --seed 42 --verbose
```

### Arguments

- `--num-games`: Number of games to collect (default: None = run until interrupted)
- `--seed`: Base seed for GameManager (default: random)
- `--num-simulations`: Number of MCTS simulations per move (default: from constants.py)
- `--output-dir`: Directory to save collected games (default: `alphago_mcts/logs/collected_games`)
- `--verbose`, `-v`: Print progress during collection

## Evaluation Script

The `evaluate_collection.py` script analyzes collected game data and provides:
- Statistical summaries (win rate, average score, etc.)
- Comparison between two collections
- Visualization plots

### Usage

#### Single Collection Evaluation

```bash
# Evaluate a single collection
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.evaluate_collection \
    --dir1 scoundrel/rl/alpha_scoundrel/alphago_mcts/logs/collected_games \
    --verbose
```

#### Compare Two Collections

```bash
# Compare AlphaGo MCTS vs Vanilla MCTS
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.evaluate_collection \
    --dir1 scoundrel/rl/alpha_scoundrel/alphago_mcts/logs/collected_games \
    --dir2 scoundrel/rl/mcts/logs/collected_games \
    --label1 "AlphaGo MCTS" \
    --label2 "Vanilla MCTS" \
    --verbose

# Compare different configurations
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.evaluate_collection \
    --dir1 ./alphago_3000_sims \
    --dir2 ./alphago_1000_sims \
    --label1 "3000 Simulations" \
    --label2 "1000 Simulations" \
    --verbose
```

### Arguments

- `--dir1`: First collection directory (required)
- `--dir2`: Second collection directory for comparison (optional)
- `--label1`: Label for first collection (default: "Collection 1")
- `--label2`: Label for second collection (default: "Collection 2")
- `--verbose`, `-v`: Print detailed progress
- `--no-plots`: Disable plot visualization

## Output

### Collected Game Format

Each game is saved as a JSON file with the following structure:

```json
{
  "metadata": {
    "seed": 12345,
    "num_simulations": 3000,
    "timestamp": "2025-12-29T10:30:00",
    "final_score": 42,
    "num_turns": 15,
    "agent_type": "alphago_mcts",
    "config": {
      "c_puct": 1.0,
      "value_weight": 0.5,
      "max_depth": 120,
      "num_workers": 8,
      "policy_large": "...",
      "policy_small": "...",
      "value_large": "..."
    }
  },
  "events": [
    {
      "turn": 0,
      "game_state": {...},
      "mcts_stats": {
        "0": {"visits": 1234, "mean_value": 0.65, "prior_prob": 0.25},
        "1": {"visits": 987, "mean_value": 0.58, "prior_prob": 0.20},
        ...
      },
      "selected_action": 0,
      "cache_stats": {"hits": 100, "misses": 50, "hit_rate": 0.67}
    },
    ...
  ]
}
```

### Evaluation Output

The evaluation script provides:

1. **Statistics Summary**:
   - Number of games
   - Win count and percentage
   - Average score (with standard deviation)
   - Best and worst scores
   - Average game length

2. **Comparison (if two directories)**:
   - Win rate difference
   - Average score difference (absolute and percentage)
   - Game length difference
   - Overall assessment

3. **Visualizations**:
   - Score distribution histograms
   - Game length distribution histograms
   - Box plots for score comparison
   - Bar charts for key metrics

## Example Workflow

```bash
# 1. Collect 100 games with AlphaGo MCTS
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.collect \
    --num-games 100 \
    --output-dir ./alphago_games \
    --verbose

# 2. Collect 100 games with Vanilla MCTS (for comparison)
python -m scoundrel.rl.mcts.collect \
    --num-games 100 \
    --output-dir ./vanilla_games \
    --verbose

# 3. Compare the two collections
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.evaluate_collection \
    --dir1 ./alphago_games \
    --dir2 ./vanilla_games \
    --label1 "AlphaGo MCTS (3000 sims)" \
    --label2 "Vanilla MCTS (100k sims)" \
    --verbose
```

## Configuration

Default parameters are defined in `constants.py`:

```python
ALPHAGO_NUM_SIMULATIONS = 3000
ALPHAGO_C_PUCT = 1.0
ALPHAGO_VALUE_WEIGHT = 0.5
ALPHAGO_MAX_DEPTH = 120
ALPHAGO_NUM_WORKERS = 8
```

You can override `num_simulations` via command-line arguments, but other parameters require modifying the constants file.

## Notes

- Collection automatically clears the agent's cache between games for consistent behavior
- Evaluation script handles missing or corrupted game files gracefully
- Plots are displayed interactively (close the window to exit)
- Large collections (1000+ games) may take time to load and visualize

