# Issue #11: Simulation Depth Limit May Be Excessive

## Priority: MEDIUM
**Estimated Impact: 2-5% speedup potential**

## Problem Description

`MCTS_MAX_DEPTH = 200` may be excessive:

- Games typically end much sooner (usually <100 steps)
- Each simulation step creates engine and applies actions
- Unnecessary computation for deep simulations

## Root Cause

- Max depth set conservatively high
- Most games end before reaching depth limit
- No early termination optimization

## Current Code Locations

- `scoundrel/rl/mcts/constants.py:11` - `MCTS_MAX_DEPTH = 200`
- `scoundrel/rl/mcts/mcts_agent.py:368` - Used in `_simulate()` loop

## Impact Analysis

- **Frequency**: Per simulation step
- **Cost**: 
  - Engine creation per step
  - Action application
  - State updates
- **Total overhead**: Moderate, but adds up

## Proposed Solutions

### Option 1: Reduce Max Depth
- Analyze typical game length
- Set depth to realistic maximum (e.g., 100)
- Still safe but faster

### Option 2: Dynamic Depth Based on Game State
- Adjust depth based on current game progress
- Shorter depth for late-game states
- Longer depth for early-game states

### Option 3: Early Termination
- Stop simulation when game_over detected
- Don't continue past terminal state
- Already partially implemented (checks `game_over`)

### Option 4: Adaptive Depth
- Start with shorter depth
- Increase if needed
- Balance exploration vs computation

## Recommended Approach

**Option 1** (Reduce Max Depth) + **Option 3** (Early Termination) - Simple and effective:
- Set realistic max depth
- Ensure early termination works correctly
- Easy to implement

## Implementation Notes

- Analyze game logs to find typical game length
- Set max depth to ~1.5x typical length for safety
- Verify early termination is working
- Test that depth limit doesn't affect game quality

## Testing Considerations

- Measure typical game lengths
- Test with reduced depth
- Verify game outcomes don't degrade
- Compare performance improvement
