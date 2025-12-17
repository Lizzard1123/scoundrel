# Issue #3: Score Calculation Overhead (MEDIUM)

## Priority: MEDIUM
**Estimated Impact: 3-8% speedup potential**

## Problem Description

`GameState.score` property recalculates score every time it's accessed:

- List comprehension: `[c for c in self.dungeon + self.room if c.type == CardType.MONSTER]`
- Reverse iteration: `next((c for c in reversed(self.discard) if c.type == CardType.POTION), None)`
- List concatenation `dungeon + room` creates new list every time
- Called once per simulation (80,000+ times per move)

## Root Cause

- Score is a property, not a cached value
- Computed on every access
- Involves list operations and iterations
- No incremental updates

## Current Code Locations

- `scoundrel/models/game_state.py:28-49` - `score` property
- `scoundrel/rl/mcts/mcts_agent.py:649` - Called in `_simulate()` for reward
- `scoundrel/rl/mcts/mcts_agent.py:259` - Called for terminal state reward

## Impact Analysis

- **Frequency**: 
  - Once per simulation (for reward)
  - Once per terminal state check
  - With 80,000 simulations: 80,000+ score calculations
- **Cost**: 
  - List comprehensions
  - List concatenation (`dungeon + room`)
  - Reverse iteration through discard
- **Total overhead**: Moderate

## Proposed Solutions

### Option 1: Cache Score in GameState (RECOMMENDED)
- Add `_cached_score` and `_score_dirty` fields
- Update cache when state changes
- Return cached value if not dirty
- Simple to implement

### Option 2: Incremental Score Updates
- Track score components incrementally:
  - `remaining_monsters_value` - update when monsters added/removed
  - `last_potion_value` - track reference, update when potion used
- Update score as state changes
- More complex but most efficient

### Option 3: Optimize Score Calculation
- Cache monster count
- Cache last potion reference
- Avoid list concatenation (iterate both lists separately)
- Reduce list operations

### Option 4: Lazy Evaluation with Memoization
- Use `@functools.lru_cache` on score method
- Cache based on state hash
- Automatic caching

## Recommended Approach

**Option 1** (Cache Score) - Best balance:
- Simple to implement
- Good performance improvement
- Low risk
- Can be combined with Option 3 for better results

## Implementation Notes

1. Add `_cached_score: Optional[int] = None` and `_score_dirty: bool = True` to GameState
2. Mark score dirty when state changes:
   - When cards picked/discarded
   - When health changes
   - When game_over changes
3. Calculate score only when dirty
4. Return cached value otherwise

## Testing Considerations

- Verify score matches current calculation
- Test all score scenarios (win, lose, exit)
- Ensure cache invalidation works correctly
- Test edge cases (empty lists, etc.)
- Compare performance improvement

## Expected Outcome

- **Performance**: 3-8% faster score calculations
- **Code**: Slightly more complex (cache management)
- **Risk**: Low (straightforward caching)
