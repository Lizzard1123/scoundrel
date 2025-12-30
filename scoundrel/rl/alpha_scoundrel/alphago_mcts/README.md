# AlphaGo-Style MCTS for Scoundrel

Combines neural networks with Monte Carlo Tree Search, inspired by AlphaGo (Nature 2016).

## Overview

This implementation enhances vanilla MCTS with three neural networks:

1. **Policy Large Network** - Strategic policy priors P(s,a) for PUCT selection
2. **Policy Small Network** - Fast rollout policy for simulations (1 FC layer)
3. **Value Large Network** - Position evaluation V(s)

### Key Improvements Over Vanilla MCTS

- **125x Fewer Simulations**: 800 vs 100,000 (neural guidance focuses search)
- **PUCT Formula**: Policy priors guide exploration instead of uniform UCB1
- **Hybrid Evaluation**: Mix value network prediction with fast rollout
- **Fast Rollouts**: PolicySmall (1 layer) is 10x faster than PolicyLarge

## Components

### 1. AlphaGo MCTS Algorithm

**Selection Phase** - Use PUCT formula:
```
PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Where:
- `Q(s,a)` = mean action value (exploitation)
- `P(s,a)` = policy prior from PolicyLarge
- `N(s,a)` = visit count for action a
- `c_puct` = exploration constant (default: 1.0)

**Evaluation Phase** - Mix value net and rollout:
```
V_final = (1 - λ) * V_value_net + λ * Z_rollout
```

Where:
- `V_value_net` = ValueLarge prediction
- `Z_rollout` = PolicySmall rollout result
- `λ` = value_weight (default: 0.5)

### 2. Neural Networks

**PolicyLarge** (Strategic Priors)
- 10 FC layers with residual connections
- Processes card embeddings + statistics
- Provides accurate policy priors P(s,a)
- Used only at expansion (not simulation bottleneck)

**PolicySmall** (Fast Rollouts)
- 1 FC layer (10x faster)
- Simple scalar features only
- 69% accuracy (sufficient for rollouts)
- Critical for simulation speed

**ValueLarge** (Position Evaluation)
- Predicts final game score
- Helps evaluate non-terminal positions
- Reduces rollout variance

## Quick Start

### Watch Agent Play

```bash
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.viewer
```

**With custom parameters:**
```bash
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.viewer \
  --num-simulations 1600 \
  --value-weight 0.3 \
  --num-workers 8
```

### Evaluate Performance

```bash
python -m scoundrel.rl.alpha_scoundrel.alphago_mcts.eval --num-games 100 --verbose
```

## Configuration

Edit `constants.py` to customize:

```python
# Model checkpoints
POLICY_LARGE_CHECKPOINT = "policy/policy_large/checkpoints/200e_70a/policy_large_epoch_200.pt"
POLICY_SMALL_CHECKPOINT = "policy/policy_small/checkpoints/140e_69a/policy_small_epoch_140.pt"
VALUE_LARGE_CHECKPOINT = "value/value_large/checkpoints/100e_193mse/value_large_epoch_100.pt"

# MCTS parameters
ALPHAGO_NUM_SIMULATIONS = 800     # Fewer needed with neural guidance
ALPHAGO_C_PUCT = 1.0               # PUCT exploration constant
ALPHAGO_VALUE_WEIGHT = 0.5         # λ: 0=pure value net, 1=pure rollout
ALPHAGO_MAX_DEPTH = 120            # Max rollout depth

# Parallelization
ALPHAGO_NUM_WORKERS = 8            # Root parallelization
```

### Tuning λ (Value Weight)

- **λ = 0.0**: Pure value network (fast, may be overconfident)
- **λ = 0.5**: Balanced mix (recommended starting point)
- **λ = 1.0**: Pure rollout (accurate but slow, like vanilla MCTS)

### Tuning c_puct

- **Lower (0.5-0.8)**: More exploitation, trust policy priors more
- **Higher (1.5-2.0)**: More exploration, diversify search more
- **Default (1.0)**: Balanced exploration/exploitation

## How It Works

### 4 MCTS Phases (AlphaGo-Style)

1. **Selection**: Traverse tree using PUCT (policy-enhanced UCB)
2. **Expansion**: Add child node, get policy priors from PolicyLarge
3. **Evaluation**: Mix ValueLarge prediction with PolicySmall rollout
4. **Backpropagation**: Propagate value up the tree

### Example Search Flow

```
Initial State
    ├─ Get P(s,·) from PolicyLarge
    │
    ├─ For 800 simulations:
    │   ├─ 1. Select: Use PUCT with P(s,a) to pick path
    │   ├─ 2. Expand: Add leaf, get P(s',·)
    │   ├─ 3. Evaluate:
    │   │   ├─ V_net ← ValueLarge(s')
    │   │   ├─ Z_rollout ← Rollout with PolicySmall
    │   │   └─ V_final = (1-λ)*V_net + λ*Z_rollout
    │   └─ 4. Backup: Propagate V_final up tree
    │
    └─ Return: Most visited child
```

## Performance

### Expected Results

Based on AlphaGo principles, we expect:

- **Win Rate**: 70%+ (vs 65% vanilla MCTS)
- **Average Score**: +20 to +25 range
- **Simulation Efficiency**: 125x fewer simulations (800 vs 100k)
- **Speed**: 0.5-1 second per move (800 sims, 8 workers)

### Key Insights to Validate

1. **Value Net Useful?** Does V(s) improve over pure rollout?
2. **Policy Priors Effective?** Does P(s,a) guide search better than UCB1?
3. **Fast Rollout Sufficient?** Is PolicySmall good enough for simulations?
4. **Optimal λ?** What mixing ratio works best for Scoundrel?

## Implementation Details

### Transposition Table

- LRU cache with 100k entries
- Caches both value net predictions and rollout results
- Critical for performance (reuse evaluations across simulations)

### Determinization

- Handles hidden information (unknown dungeon cards)
- Shuffles unknown cards, keeps known cards fixed
- Each simulation uses different determinization

### Parallelization

- Root parallelization: each worker runs independent simulations
- Aggregate statistics at the end
- 8 workers typical on modern CPUs

## Files

```
alphago_mcts/
├── __init__.py               # Module exports
├── README.md                 # This file
├── PLAN.md                   # Implementation plan (reference)
├── constants.py              # Configuration parameters
├── alphago_node.py           # MCTS node with PUCT
├── alphago_agent.py          # Main agent implementation
├── eval.py                   # Evaluation script
├── viewer.py                 # Interactive viewer
├── checkpoints/              # Model checkpoints (not included)
└── logs/                     # Evaluation results
```

## Usage Examples

### Programmatic Usage

```python
from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_agent import AlphaGoAgent
from scoundrel.game.game_manager import GameManager

# Create agent
agent = AlphaGoAgent(
    num_simulations=800,
    c_puct=1.0,
    value_weight=0.5,
    num_workers=8
)

# Play game
engine = GameManager(seed=42)
state = engine.restart()

while not state.game_over:
    action_idx = agent.select_action(state)
    action_enum = agent.translator.decode_action(action_idx)
    engine.execute_turn(action_enum)
    state = engine.get_state()

print(f"Final score: {state.score}")
```

### Custom Model Checkpoints

```python
agent = AlphaGoAgent(
    policy_large_checkpoint="path/to/policy_large.pt",
    policy_small_checkpoint="path/to/policy_small.pt",
    value_checkpoint="path/to/value_large.pt",
    num_simulations=1600,
    value_weight=0.3
)
```

## Troubleshooting

### Agent runs slowly
- Reduce `num_simulations` (try 400 or 200)
- Increase `num_workers` for parallelization
- Increase `value_weight` (λ) to rely more on fast value net

### Agent underperforms
- Try different `value_weight` (0.3-0.7)
- Tune `c_puct` (0.8-1.5)
- Check model checkpoint paths are correct
- Ensure models are properly trained

### Out of memory
- Reduce `ALPHAGO_TRANSPOSITION_TABLE_SIZE` in constants.py
- Use device="cpu" instead of "mps" or "cuda"

## References

### Papers
- **AlphaGo (Nature 2016)**: Silver et al., "Mastering the game of Go with deep neural networks and tree search"
- **PUCT Algorithm**: Rosin (2011), "Multi-armed bandits with episode context"
- **AlphaZero (Science 2018)**: Silver et al., "A general reinforcement learning algorithm"

### Internal Codebase
- `scoundrel/rl/mcts/` - Vanilla MCTS implementation
- `scoundrel/rl/alpha_scoundrel/policy/` - Policy networks
- `scoundrel/rl/alpha_scoundrel/value/` - Value network
- `scoundrel/rl/translator.py` - State encoding
- `scoundrel/game/game_logic.py` - Game state transitions

## Contributing

When modifying this implementation:

1. **Maintain PUCT formula** - Core to AlphaGo approach
2. **Profile performance** - Rollout policy is bottleneck
3. **Test incrementally** - Each phase should be testable
4. **Document insights** - What works, what doesn't, why

## License

Part of the Scoundrel project.

---

*Last updated: 2025-12-29*

