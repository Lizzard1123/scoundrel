# Issue #5: Translator Overhead (Action Mask Creation)

## Priority: HIGH
**Estimated Impact: 3-8% speedup potential**

## Problem Description

`get_action_mask()` creates a new PyTorch tensor every time it's called:

- Called frequently in hot paths:
  - `_get_valid_actions()` (line 314)
  - `_random_policy()` (line 400)
  - `_heuristic_policy()` (line 408)
  - During simulation rollout (line 370)
- Tensor creation has overhead
- Only needs boolean list, not necessarily a tensor

## Root Cause

- Translator designed for neural network training (needs tensors)
- MCTS doesn't need tensors, just boolean lists
- Creating tensors adds unnecessary overhead

## Current Code Locations

- `scoundrel/rl/translator.py:86-101` - `get_action_mask()` method
- `scoundrel/rl/mcts/mcts_agent.py:314` - Called in `_get_valid_actions()`

## Impact Analysis

- **Frequency**: Called multiple times per simulation:
  - Once in `_get_valid_actions()` 
  - Once per step in simulation rollout
  - With 10,000 simulations and ~50 steps average: ~500,000 tensor creations
- **Cost**: 
  - Tensor allocation overhead
  - Memory allocation
  - Type conversion
- **Total overhead**: Moderate but unnecessary

## Proposed Solutions

### Option 1: Cache Action Masks
- Cache action mask per state hash
- Reuse mask for same state
- Only create new mask when state changes

### Option 2: Return List Instead of Tensor
- Create lightweight `get_action_mask_list()` method
- Returns Python list instead of tensor
- MCTS can use list directly

### Option 3: Inline Action Validation
- Skip mask creation entirely
- Check action validity inline
- Use simple list comprehension: `[i for i in range(5) if is_valid(i)]`

### Option 4: Pre-compute Valid Actions
- Store valid actions in GameState
- Update when state changes
- No need to compute each time

## Recommended Approach

**Option 3** (Inline Action Validation) - Simplest and fastest:
- No object creation needed
- Direct validation checks
- Eliminates translator dependency for MCTS

## Implementation Notes

- Move validation logic to MCTS agent
- Check `len(game_state.room)` for pick actions
- Check `game_state.can_avoid` for avoid action
- Simple list comprehension is fastest

## Testing Considerations

- Verify all valid actions are detected correctly
- Test edge cases (empty room, can't avoid, etc.)
- Ensure same behavior as current implementation
