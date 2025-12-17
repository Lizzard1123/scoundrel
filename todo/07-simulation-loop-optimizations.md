# Issue #7: Simulation Loop Micro-Optimizations (LOW-MEDIUM)

## Priority: LOW-MEDIUM
**Estimated Impact: 2-5% speedup potential**

## Problem Description

After eliminating engine usage (Issue #1), there are still micro-optimizations possible in the simulation loop:

- Multiple `game_over` checks (redundant)
- `_get_valid_actions()` called every iteration
- Action enum decoding overhead
- Loop structure can be optimized

## Root Cause

- Defensive programming (multiple checks)
- Function call overhead
- Redundant computations

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:595-652` - `_simulate()` method
- After Issue #1 fix, this will be the optimized version

## Impact Analysis

- **Frequency**: 
  - Once per simulation step (avg 50 steps × 80,000 simulations = 4M iterations)
  - Small overheads add up
- **Cost**: 
  - Redundant `game_over` checks
  - Function call overhead
  - Enum decoding
- **Total overhead**: Small but measurable

## Proposed Solutions

### Option 1: Optimize Loop Structure (RECOMMENDED)
- Single `game_over` check per iteration
- Cache valid_actions when possible
- Inline simple operations
- Reduce function call overhead

### Option 2: Early Exit Optimizations
- Check `game_over` before getting valid actions
- Skip unnecessary computations
- More aggressive early termination

### Option 3: Cache Action Enums
- Pre-compute action enum mappings
- Avoid repeated `decode_action()` calls
- Simple optimization

## Recommended Approach

**Option 1** (Optimize Loop Structure) - Best balance:
- Simple optimizations
- Good performance improvement
- Low risk

## Implementation Notes

1. After Issue #1 fix, optimize loop:
   ```python
   current_state = game_state
   depth = 0
   while depth < self.max_depth and not current_state.game_over:
       valid_actions = self._get_valid_actions(current_state)
       if not valid_actions:
           break
       action = self._random_policy(current_state, valid_actions)
       action_enum = self.translator.decode_action(action)
       current_state = self._apply_action_to_state(current_state, action_enum)
       depth += 1
   ```

2. Combine `game_over` check with loop condition
3. Cache action enums if beneficial
4. Reduce redundant checks

## Testing Considerations

- Verify simulation produces identical results
- Test edge cases (terminal states)
- Compare performance improvement
- Ensure no regressions

## Expected Outcome

- **Performance**: 2-5% faster simulation loop
- **Code**: Slightly cleaner
- **Risk**: Low (micro-optimizations)

## Note

This optimization should be done **after** Issue #1 (eliminating engine), as the loop structure will change significantly.
