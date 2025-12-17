# Issue #10: Determinization Inefficiency

## Priority: HIGH
**Estimated Impact: 3-8% speedup potential**

## Problem Description

`_determinize_state()` always deep copies even when determinization isn't needed:

- Deep copies state even when `number_avoided == 0` (no shuffling needed)
- List slicing creates new lists: `[:unknown_count]`, `[unknown_count:]`
- Reconstructs dungeon list even when no changes

## Root Cause

- Determinization always copies state first
- Doesn't check if determinization is actually needed
- List operations create unnecessary copies

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:463-488` - `_determinize_state()` method
- `scoundrel/rl/mcts/mcts_agent.py:146` - Called in `_sequential_search()`

## Impact Analysis

- **Frequency**: Once per simulation (10,000+ times per move)
- **Cost**: 
  - Deep copy even when not needed
  - List slicing operations
  - List reconstruction
- **Total overhead**: Moderate, but unnecessary

## Proposed Solutions

### Option 1: Early Return for No Determinization
- Check `number_avoided == 0` first
- Return original state (or shallow copy) if no determinization needed
- Only deep copy when actually needed

### Option 2: Optimize List Operations
- Use `random.shuffle()` in-place on slice
- Avoid list reconstruction
- More efficient list manipulation

### Option 3: Lazy Determinization
- Only determinize when actually needed
- Cache determinized states
- Reuse when possible

### Option 4: Shallow Copy Optimization
- Use shallow copy instead of deep copy
- Only copy lists that need shuffling
- Cards are immutable, so shallow copy may suffice

## Recommended Approach

**Option 1** (Early Return) + **Option 4** (Shallow Copy) - Best combination:
- Skip determinization when not needed
- Use shallow copy when determinization is needed
- Cards are immutable, so safe

## Implementation Notes

- Check `number_avoided == 0` first
- Return state reference or shallow copy
- Only deep copy when shuffling is needed
- Consider if shallow copy is safe (cards are immutable)

## Testing Considerations

- Verify determinization still works correctly
- Test with `number_avoided == 0` (should be fast)
- Test with `number_avoided > 0` (should shuffle correctly)
- Ensure no state leakage between simulations
