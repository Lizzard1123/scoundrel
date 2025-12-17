# Issue #1: Simulation Phase Engine Usage (CRITICAL)

## Priority: CRITICAL
**Estimated Impact: 30-50% speedup potential**

## Problem Description

The `_simulate()` method creates a `GameManager` engine and uses `engine.execute_turn()` in a loop, which is highly inefficient:

- **Line 615**: Creates engine: `engine = self._create_engine_from_state(game_state)`
- **Line 638**: Calls `engine.execute_turn(action_enum)` in simulation loop
- Each `execute_turn()` call goes through `GameManager` logic with UI-specific overhead
- Engine creation involves state copying: `GameManager.from_state(game_state.copy())`
- With 80,000 simulations × avg 50 steps per simulation = **~4 million engine operations per move**

## Root Cause

- `_simulate()` still uses the old pattern of creating engines
- `_select()` and `_expand()` already use the optimized `apply_action_to_state()` pure function
- Simulation phase is the most frequent operation (happens once per simulation)
- `execute_turn()` has unnecessary overhead (command_text, UI validation, etc.)

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:615` - `_create_engine_from_state()` call
- `scoundrel/rl/mcts/mcts_agent.py:638` - `engine.execute_turn()` in loop
- `scoundrel/rl/mcts/mcts_agent.py:639` - `engine.get_state()` call

## Impact Analysis

- **Frequency**: Once per simulation (80,000+ times per move)
- **Cost**: 
  - Engine creation (state copy + GameManager initialization)
  - `execute_turn()` overhead (UI logic, validation)
  - `get_state()` calls every iteration
- **Total overhead**: Massive - likely the #1 remaining performance bottleneck

## Proposed Solutions

### Option 1: Replace Engine with Pure Function (RECOMMENDED)
- Replace `engine.execute_turn()` with direct `apply_action_to_state()` calls
- Matches the pattern already used in `_select()` and `_expand()`
- Eliminates all engine creation overhead
- Simplifies simulation loop

### Option 2: Reuse Single Engine Instance
- Create one engine at start of simulation
- Reuse it by resetting state instead of creating new
- Still has `execute_turn()` overhead

### Option 3: Engine Pool
- Maintain pool of pre-initialized engines
- Reuse from pool instead of creating new
- Still has `execute_turn()` overhead

## Recommended Approach

**Option 1** (Pure Function) - Best performance:
- Eliminates engine creation entirely
- Pure function is fastest
- Matches existing optimized pattern
- Low risk (already proven in selection/expansion)

## Implementation Notes

1. Replace simulation loop to use `apply_action_to_state()`:
   ```python
   current_state = game_state
   while depth < self.max_depth:
       if current_state.game_over:
           break
       valid_actions = self._get_valid_actions(current_state)
       if not valid_actions:
           break
       action = self._random_policy(current_state, valid_actions)
       action_enum = self.translator.decode_action(action)
       current_state = self._apply_action_to_state(current_state, action_enum)
       depth += 1
   ```

2. Remove `_create_engine_from_state()` call from `_simulate()`
3. Remove `engine.get_state()` calls
4. Can potentially remove `_create_engine_from_state()` entirely if not used elsewhere

## Testing Considerations

- Verify simulation produces identical results
- Test all action types (pick card, avoid)
- Ensure no state leakage between simulations
- Compare performance improvement
- Verify transposition table still works correctly

## Expected Outcome

- **Performance**: 30-50% faster simulations
- **Code**: Simpler, more consistent with rest of codebase
- **Risk**: Low (pattern already proven)
