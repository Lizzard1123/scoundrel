# Issue #3: Repeated Engine Creation in Selection/Expansion

## Priority: CRITICAL
**Estimated Impact: 10-20% speedup potential**

## Problem Description

During tree traversal (`_select()` and `_expand()`), a new engine is created for **every action application**:

- `_select()` (line 329): Creates engine for each step down the tree
- `_expand()` (line 348): Creates engine to apply action
- Each engine creation involves deep copying state

## Root Cause

- Current design treats each action application as independent
- No reuse of engine instances
- Each `_create_engine_from_state()` call does a deep copy

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:317-333` - `_select()` method
- `scoundrel/rl/mcts/mcts_agent.py:335-357` - `_expand()` method
- `scoundrel/rl/mcts/mcts_agent.py:490-494` - `_create_engine_from_state()`

## Impact Analysis

- **Frequency**: 
  - Selection: Creates engine for each level traversed (typically 1-10 levels)
  - Expansion: Creates engine once per expansion
  - Per simulation: ~2-11 engine creations
  - With 10,000 simulations: ~20,000-110,000 engine creations per move
- **Cost**: Each creation = deep copy + GameManager initialization
- **Total overhead**: Massive multiplier effect

## Proposed Solutions

### Option 1: Reuse Single Engine Instance
- Create one engine at start of simulation
- Reuse it by resetting state instead of creating new
- Apply actions sequentially on same engine

### Option 2: State-Based Action Application
- Extract action application logic to pure function
- `apply_action(state, action) -> new_state`
- No engine needed, just state transformation

### Option 3: Engine Pool
- Maintain pool of pre-initialized engines
- Reuse from pool instead of creating new
- Reset state when returning to pool

### Option 4: Incremental State Updates
- Track state changes incrementally
- Only copy what changed
- Avoid full state recreation

## Recommended Approach

**Option 2** (State-Based Action Application) - Most efficient:
- Eliminates engine creation entirely
- Pure function is fastest
- Matches functional programming paradigm
- Can be easily cached/memoized

## Implementation Notes

- Need to extract `execute_turn()` logic into pure function
- Must handle all game rules correctly:
  - Card picking
  - Combat calculations
  - Room drawing
  - Avoid mechanics
- Ensure deterministic behavior

## Testing Considerations

- Verify action application produces identical results
- Test all action types
- Ensure no state leakage between simulations
- Compare results with current implementation
