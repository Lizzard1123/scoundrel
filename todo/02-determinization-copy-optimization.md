# Issue #2: Determinization Copy Optimization (HIGH)

## Priority: HIGH
**Estimated Impact: 5-10% speedup potential**

## Problem Description

`_determinize_state()` always copies state even when determinization isn't needed:

- **Line 763**: Early return still calls `game_state.copy()` when `number_avoided == 0`
- Copy happens before checking if determinization is actually needed
- With 80,000 simulations, even small overhead adds up significantly

## Root Cause

- Determinization always copies state first
- Doesn't check if determinization is actually needed before copying
- When `number_avoided == 0`, no shuffling is needed, but copy still happens

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:750-788` - `_determinize_state()` method
- `scoundrel/rl/mcts/mcts_agent.py:763` - Early return with unnecessary copy

## Impact Analysis

- **Frequency**: Once per simulation (80,000+ times per move)
- **Cost**: 
  - State copy even when not needed (`number_avoided == 0`)
  - List operations for copying all state lists
- **Total overhead**: Moderate but unnecessary

## Proposed Solutions

### Option 1: Skip Copy When Not Needed (RECOMMENDED)
- Check `number_avoided == 0` first
- Return original state reference if determinization not needed
- Only copy when actually shuffling
- **Note**: Need to verify this is safe (state immutability)

### Option 2: Lazy Copy
- Only copy when determinization is actually needed
- Use shallow copy (already optimized in `GameState.copy()`)
- Copy only the dungeon list that needs shuffling

### Option 3: Conditional Copy
- Copy state only when `number_avoided > 0`
- Return reference otherwise
- Ensure no mutation issues

## Recommended Approach

**Option 1** (Skip Copy When Not Needed) - Best performance:
- Eliminates unnecessary copies
- Simple to implement
- Need to verify state immutability is safe

## Implementation Notes

1. Check `number_avoided == 0` first before any copying
2. If no determinization needed, return original state (or verify if copy is required for safety)
3. Only copy when actually shuffling cards
4. Consider if determinization can mutate state (shouldn't, but verify)

## Testing Considerations

- Verify determinization still works correctly
- Test with `number_avoided == 0` (should be fast)
- Test with `number_avoided > 0` (should shuffle correctly)
- Ensure no state leakage between simulations
- Verify parallel workers don't share state accidentally

## Expected Outcome

- **Performance**: 5-10% faster determinization
- **Code**: Slightly simpler
- **Risk**: Low-Medium (need to verify state immutability)
