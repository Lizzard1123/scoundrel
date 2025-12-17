# Issue #8: No State Reuse/Caching

## Priority: MEDIUM
**Estimated Impact: 5-15% speedup potential**

## Problem Description

Same game states may be reached multiple times but are recalculated:

- No transposition table
- No state caching
- Each simulation starts fresh
- Same states evaluated repeatedly

## Root Cause

- MCTS doesn't currently cache state evaluations
- Each simulation is independent
- No reuse of previously computed results

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:130-161` - `_sequential_search()` method
- `scoundrel/rl/mcts/mcts_agent.py:359-387` - `_simulate()` method

## Impact Analysis

- **Frequency**: Depends on game structure
- **Cost**: 
  - Repeated state evaluations
  - Redundant simulations
  - Missed opportunity for optimization
- **Total overhead**: Moderate, but could be significant

## Proposed Solutions

### Option 1: Transposition Table
- Cache simulation results by state hash
- Reuse results for identical states
- Trade memory for computation

### Option 2: State Pooling
- Maintain pool of common states
- Reuse state objects instead of creating new
- Reduce allocation overhead

### Option 3: Incremental Updates
- Track which states have been evaluated
- Skip re-evaluation of known states
- Use cached results

### Option 4: Tree Reuse Between Moves
- Keep tree structure between moves
- Reuse subtrees that are still valid
- More complex but potentially powerful

## Recommended Approach

**Option 1** (Transposition Table) - Most effective:
- Can significantly reduce redundant work
- Standard MCTS optimization
- Good memory/computation tradeoff

## Implementation Notes

- Use state hash as key
- Store: visits, value, best action
- Limit table size (LRU eviction)
- Handle determinization correctly

## Testing Considerations

- Verify cached results are correct
- Test memory usage
- Measure cache hit rate
- Compare performance improvement
