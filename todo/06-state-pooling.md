# Issue #6: State Pooling (MEDIUM)

## Priority: MEDIUM
**Estimated Impact: 5-10% speedup potential**

## Problem Description

GameState objects are created and destroyed frequently:

- Each simulation creates multiple GameState copies
- State copying happens frequently (determinization, action application)
- GC pressure from frequent allocations/deallocations
- With 80,000 simulations, millions of GameState objects created per move

## Root Cause

- No reuse of GameState objects
- Each copy creates new object
- Frequent allocations increase GC pressure
- Memory churn affects performance

## Current Code Locations

- `scoundrel/models/game_state.py:67-89` - `copy()` method
- `scoundrel/rl/mcts/mcts_agent.py:750-788` - `_determinize_state()` creates copies
- `scoundrel/game/game_logic.py:28` - `apply_action_to_state()` creates copies

## Impact Analysis

- **Frequency**: 
  - Once per determinization (80,000+ times)
  - Multiple times per simulation (action applications)
  - Millions of GameState objects created per move
- **Cost**: 
  - Object allocation overhead
  - GC pressure
  - Memory churn
- **Total overhead**: Moderate but could be significant

## Proposed Solutions

### Option 1: State Pool (RECOMMENDED)
- Maintain pool of pre-allocated GameState objects
- Reuse states from pool instead of creating new
- Reset state fields when returning to pool
- Reduces allocation overhead

### Option 2: In-Place State Updates
- Modify state in place where safe
- Only copy when necessary
- More complex state management

### Option 3: Object Pool Library
- Use existing object pool library
- More robust but adds dependency

### Option 4: Keep Current
- Current approach may be acceptable
- GC is generally efficient
- Simpler code

## Recommended Approach

**Option 1** (State Pool) - Best performance:
- Reduces allocation overhead
- Reduces GC pressure
- Can be implemented incrementally

## Implementation Notes

1. Create `StatePool` class:
   ```python
   class StatePool:
       def __init__(self, initial_size=100):
           self.pool = [GameState() for _ in range(initial_size)]
           self.available = list(range(initial_size))
       
       def acquire(self) -> GameState:
           if self.available:
               idx = self.available.pop()
               return self.pool[idx]
           return GameState()  # Fallback
       
       def release(self, state: GameState):
           # Reset state
           state.dungeon.clear()
           state.room.clear()
           # ... reset all fields
           # Add back to pool
   ```

2. Use pool in MCTS agent
3. Reset state when returning to pool
4. Handle pool exhaustion (create new or expand pool)

## Testing Considerations

- Verify state pooling doesn't cause state leakage
- Test pool exhaustion scenarios
- Measure memory usage
- Compare performance improvement
- Ensure thread safety if using in parallel mode

## Expected Outcome

- **Performance**: 5-10% faster (reduced allocation overhead)
- **Code**: More complex (pool management)
- **Risk**: Medium (need careful state management)
