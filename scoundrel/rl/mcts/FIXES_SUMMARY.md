# MCTS Implementation Fixes - Summary

## What Was Wrong

Your MCTS implementation was performing worse than random because of a **critical state management bug**. The game state was not being properly threaded through the MCTS phases, causing the algorithm to:

1. Build a tree based on one state
2. But simulate from a completely different state
3. Essentially making random decisions with extra overhead

## Critical Fix: State Threading

### Before (Broken):
```python
def select_action(self, game_state: GameState) -> int:
    root = self._create_node(game_state)
    
    for _ in range(self.num_simulations):
        node = self._select(root, game_state)          # ❌ Uses original state
        node = self._expand(node, game_state)          # ❌ Uses original state
        reward = self._simulate(node, game_state)      # ❌ Uses original state
        self._backpropagate(node, reward)
```

**Problem:** Every phase gets the same `game_state`, so the tree exploration doesn't actually follow the game's state transitions!

### After (Fixed):
```python
def select_action(self, game_state: GameState) -> int:
    root = self._create_node(game_state)
    
    for _ in range(self.num_simulations):
        simulation_state = copy.deepcopy(game_state)   # ✓ Fresh copy
        node, simulation_state = self._select(root, simulation_state)    # ✓ Returns updated state
        node, simulation_state = self._expand(node, simulation_state)    # ✓ Uses and updates state
        reward = self._simulate(simulation_state)                        # ✓ Simulates from correct state
        self._backpropagate(node, reward)
```

**Fix:** Each phase receives the state from the previous phase and returns the updated state!

## Other Important Fixes

### 2. Intelligent Heuristic Policy

Replaced random rollouts with a heuristic that understands Scoundrel:

```python
def _heuristic_policy(self, game_state: GameState) -> int:
    # ✓ Prioritizes healing when low on health
    # ✓ Values weapons based on strength
    # ✓ Avoids fatal monster encounters
    # ✓ Calculates actual damage using Combat system
    # ✓ Considers weapon usage rules (ascending values)
    # ✓ Evaluates when to avoid rooms
```

### 3. Terminal State Tracking

Added `is_game_over` flag to nodes to properly identify terminal states.

### 4. Edge Case Handling

Added safety check for when no simulations complete successfully.

## Files Changed

1. **`mcts_agent.py`**
   - Fixed state threading in `select_action()`
   - Updated `_select()` to return `(node, state)` tuple
   - Updated `_expand()` to return `(node, state)` tuple  
   - Updated `_simulate()` to only take state parameter
   - Rewrote `_heuristic_policy()` with Scoundrel-specific logic

2. **`mcts_node.py`**
   - Added `is_game_over` field
   - Removed incorrect `is_terminal()` method

3. **`constants.py`**
   - Changed default to `USE_RANDOM_ROLLOUT = False`

## How to Test

### Quick Test (10 games, 50 simulations):
```bash
python -m scoundrel.rl.mcts.mcts --num-simulations 50 --num-games 10
```

### Full Evaluation (100 games, 100 simulations):
```bash
python -m scoundrel.rl.mcts.mcts --num-simulations 100 --num-games 100
```

### Watch it Play:
```bash
python -m scoundrel.rl.mcts.viewer --num-simulations 100
```

## Expected Results

**Before fixes:**
- Average score: Negative (worse than random)
- Win rate: Very low
- Behavior: Seemingly random, no learning from simulations

**After fixes:**
- Average score: Should be positive and improving with more simulations
- Win rate: Should be significantly better than random
- Behavior: Should make sensible decisions (take weapons, avoid fatal damage, heal when needed)

## Why It's Better Now

1. **Proper Tree Search:** MCTS now actually explores the game tree correctly
2. **Meaningful Simulations:** Rollouts use domain knowledge instead of random moves
3. **Correct Evaluation:** Rewards are backpropagated from the right states
4. **Smart Decisions:** Heuristic understands game mechanics (weapons, healing, combat)

## Performance Tuning

If you want to improve performance further:

```python
# In constants.py

# More simulations = better play (but slower)
MCTS_NUM_SIMULATIONS = 200  # Try 200-500 for strong play

# Higher exploration = more diverse search
MCTS_EXPLORATION_CONSTANT = 2.0  # Try 1.0-2.0

# For faster testing
MCTS_NUM_SIMULATIONS = 50
```

## Comparison with Random Play

To verify the fix, you can compare:

1. **Random agent:** Just picks random valid actions
2. **MCTS (fixed):** Should significantly outperform random
3. **MCTS with more sims:** Should improve with 200-500 simulations

The key metric is **average score** - positive scores mean the agent survived, higher is better.



