# Issue #6: Heuristic Policy Computation Overhead

## Priority: MEDIUM
**Estimated Impact: 2-5% speedup potential (when heuristic enabled)**

## Problem Description

`_heuristic_policy()` performs significant computation during simulation rollouts:

- Evaluates all valid actions
- Calls `Combat.can_use_weapon()` for each monster
- Calls `Combat.calculate_damage()` for damage calculation
- Multiple list comprehensions and conditionals
- Only used when `use_random_rollout=False`

## Root Cause

- Heuristic tries to be smart but adds computational cost
- Called once per simulation step
- With max_depth=200, this adds up quickly

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:403-461` - `_heuristic_policy()` method
- `scoundrel/rl/mcts/mcts_agent.py:378` - Called in `_simulate()` when not using random

## Impact Analysis

- **Frequency**: Only when `USE_RANDOM_ROLLOUT=False`
- **Cost**: 
  - Combat calculations per action evaluation
  - List iterations
  - Conditional logic
- **Total overhead**: Moderate, but only affects heuristic mode

## Proposed Solutions

### Option 1: Optimize Heuristic Logic
- Cache combat calculations
- Early exit conditions
- Simplify scoring logic

### Option 2: Use Random Rollout (Current Default)
- Random is faster and often just as good
- Already the default (`USE_RANDOM_ROLLOUT=True`)
- Consider removing heuristic entirely

### Option 3: Lighter Heuristic
- Simplified heuristic with fewer calculations
- Only evaluate top N actions
- Use approximate scoring

### Option 4: Hybrid Approach
- Use heuristic for first few steps
- Switch to random for rest of rollout
- Best of both worlds

## Recommended Approach

**Option 2** (Use Random Rollout) - Already optimal:
- Current default is already random
- Heuristic may not provide enough benefit to justify cost
- Can be removed or kept as optional

## Implementation Notes

- If keeping heuristic, optimize combat calculations
- Consider memoizing `Combat.calculate_damage()` results
- Simplify scoring function

## Testing Considerations

- Compare game outcomes with random vs heuristic
- Measure actual performance difference
- Verify heuristic doesn't significantly improve play quality
