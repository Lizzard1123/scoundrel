# Issue #9: Score Calculation Overhead

## Priority: MEDIUM
**Estimated Impact: 2-5% speedup potential**

## Problem Description

`GameState.score` property performs computation every time it's accessed:

- List comprehension to find remaining monsters: `[c for c in self.dungeon + self.room if c.type == CardType.MONSTER]`
- Iteration through discard to find last potion: `next((c for c in reversed(self.discard) if c.type == CardType.POTION), None)`
- Called frequently during backpropagation and simulation

## Root Cause

- Score is a property, not a cached value
- Computed on every access
- Involves list operations and iterations

## Current Code Locations

- `scoundrel/models/game_state.py:28-49` - `score` property
- `scoundrel/rl/mcts/mcts_agent.py:387` - Called in `_simulate()` for reward

## Impact Analysis

- **Frequency**: 
  - Once per simulation (for reward)
  - Potentially during backpropagation
  - With 10,000 simulations: 10,000+ score calculations
- **Cost**: 
  - List comprehensions
  - List concatenation (`dungeon + room`)
  - Reverse iteration through discard
- **Total overhead**: Moderate

## Proposed Solutions

### Option 1: Cache Score in GameState
- Add `_cached_score` field
- Update when state changes
- Return cached value

### Option 2: Incremental Score Updates
- Track score components incrementally
- Update score when actions are applied
- No recalculation needed

### Option 3: Optimize Score Calculation
- Cache monster count
- Cache last potion reference
- Reduce list operations

### Option 4: Lazy Evaluation with Memoization
- Use `@functools.lru_cache` on score method
- Cache based on state hash
- Automatic caching

## Recommended Approach

**Option 2** (Incremental Score Updates) - Most efficient:
- No recalculation needed
- Update score as state changes
- Fastest approach

## Implementation Notes

- Track: health, remaining_monsters_value, last_potion_value
- Update when cards are picked/discarded
- Update when health changes
- Handle edge cases (game over, exit)

## Testing Considerations

- Verify score matches current calculation
- Test all score scenarios (win, lose, exit)
- Ensure incremental updates are correct
- Compare performance improvement
