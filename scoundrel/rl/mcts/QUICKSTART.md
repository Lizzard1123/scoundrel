# MCTS Quick Start Guide

## Installation

Make sure you have the environment set up:

```bash
conda env create -f environment.yml
conda activate scoundrel
```

## 1. Watch MCTS Play (Interactive Viewer)

The fastest way to see MCTS in action:

```bash
# From project root
python -m scoundrel.rl.mcts.viewer
```

**Controls:**
- Press `Space` or `Enter` to execute the next action
- Press `r` to restart the game
- Press `q` to quit

**Adjust simulation count:**
```bash
# More simulations = better play, but slower
python -m scoundrel.rl.mcts.viewer --num-simulations 200
```

## 2. Run Batch Evaluation

Evaluate MCTS performance over multiple games:

```bash
# Using the shell script (recommended)
./scoundrel/rl/mcts/scripts/train.sh

# Or directly with Python
python -m scoundrel.rl.mcts.mcts
```

**Custom parameters:**
```bash
# 50 simulations per move, 100 games total
python -m scoundrel.rl.mcts.mcts --num-simulations 50 --num-games 100

# Quiet mode (less output)
python -m scoundrel.rl.mcts.mcts --quiet

# Save statistics every 50 games
python -m scoundrel.rl.mcts.mcts --save-interval 50
```

## 3. View Statistics

After running batch evaluation, check the results:

```bash
# Statistics are saved to:
cat scoundrel/rl/mcts/logs/mcts_stats.json
```

The JSON file contains:
- Average score across all games
- Best and worst scores
- Win rate (games where health > 0)
- Full score history
- Moving averages

## 4. Configuration

Edit `scoundrel/rl/mcts/constants.py` to change defaults:

```python
# Number of simulations per move
MCTS_NUM_SIMULATIONS = 100

# UCB1 exploration constant (sqrt(2) is standard)
MCTS_EXPLORATION_CONSTANT = 1.414

# Maximum depth for rollout simulations
MCTS_MAX_DEPTH = 200

# Use random vs heuristic rollout policy
USE_RANDOM_ROLLOUT = True
```

## 5. Programmatic Usage

Use MCTS in your own code:

```python
from scoundrel.game.game_manager import GameManager
from scoundrel.rl.mcts.mcts_agent import MCTSAgent

# Create agent and game
agent = MCTSAgent(num_simulations=100)
engine = GameManager()
state = engine.restart()

# Play one turn
action_idx = agent.select_action(state)
action_enum = agent.translator.decode_action(action_idx)
engine.execute_turn(action_enum)
```

## 6. Comparing with Transformer PPO

Run both methods and compare:

```bash
# Run MCTS
python -m scoundrel.rl.mcts.mcts --num-games 100

# Run Transformer PPO (if trained)
python -m scoundrel.rl.transformer_mlp.viewer \
  --checkpoint scoundrel/rl/transformer_mlp/checkpoints/ppo_latest.pt
```

**Key Differences:**
- MCTS: No training, slower per move, deterministic
- PPO: Requires training, fast per move, learned policy

## 7. Troubleshooting

**MCTS is too slow:**
- Reduce `--num-simulations` (e.g., 50 instead of 100)
- Check `MCTS_MAX_DEPTH` in constants.py

**Poor performance:**
- Increase `--num-simulations` (e.g., 200 or 500)
- Try adjusting `MCTS_EXPLORATION_CONSTANT`
- Experiment with `USE_RANDOM_ROLLOUT = False` for heuristic rollouts

**Import errors:**
- Make sure you're in the conda environment: `conda activate scoundrel`
- Install the package: `pip install -e .`

## 8. Clean Up

Remove logs and start fresh:

```bash
./scoundrel/rl/mcts/scripts/train.sh --clear
```

## Performance Tips

### Fast Testing (Quick Results)
```bash
python -m scoundrel.rl.mcts.mcts \
  --num-simulations 25 \
  --num-games 10 \
  --quiet
```

### High Quality Play (Slow but Better)
```bash
python -m scoundrel.rl.mcts.mcts \
  --num-simulations 500 \
  --num-games 100
```

### Balanced (Recommended)
```bash
python -m scoundrel.rl.mcts.mcts \
  --num-simulations 100 \
  --num-games 100
```

## Next Steps

1. Run the viewer to understand how MCTS plays
2. Run batch evaluation to get performance statistics
3. Experiment with different simulation counts
4. Compare with other RL methods in `scoundrel/rl/`
5. Modify the rollout policy in `mcts_agent.py` for better performance

For more details, see [README.md](README.md) and the main RL documentation at `scoundrel/rl/README.md`.



