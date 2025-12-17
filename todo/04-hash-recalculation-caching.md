# Issue #4: Hash Recalculation Caching (MEDIUM)

## Priority: MEDIUM
**Estimated Impact: 2-5% speedup potential**

## Problem Description

`_hash_state()` recalculates hash even when state hasn't changed:

- Hash calculation iterates through cards and creates tuples
- Called multiple times per simulation:
  - For transposition table lookup
  - For node creation
  - For caching
- Hash is deterministic based on state, so can be cached

## Root Cause

- Hash is computed every time `_hash_state()` is called
- No caching mechanism
- State hash doesn't change unless state changes
- Multiple calls with same state recalculate unnecessarily

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:482-514` - `_hash_state()` method
- Called from:
  - `_create_node()` (line 471)
  - `_simulate()` for transposition table (line 603)
  - Terminal state reward (line 254)

## Impact Analysis

- **Frequency**: 
  - Multiple times per simulation
  - Once per node creation
  - Once per transposition table lookup
  - With 80,000 simulations: potentially 200,000+ hash calculations
- **Cost**: 
  - Card iteration
  - Tuple creation
  - Hash computation
- **Total overhead**: Moderate

## Proposed Solutions

### Option 1: Cache Hash in GameState (RECOMMENDED)
- Add `_cached_hash: Optional[int] = None` and `_hash_dirty: bool = True` to GameState
- Calculate hash only when dirty
- Invalidate hash when state changes
- Return cached value otherwise

### Option 2: Lazy Hash Property
- Make hash a property on GameState
- Cache result after first calculation
- Invalidate on state mutation

### Option 3: Hash Only When Needed
- Only compute hash when actually needed (transposition table, node creation)
- Skip hash computation in some paths
- More complex logic

## Recommended Approach

**Option 1** (Cache Hash in GameState) - Best balance:
- Simple to implement
- Good performance improvement
- Consistent with score caching approach
- Low risk

## Implementation Notes

1. Add `_cached_hash: Optional[int] = None` and `_hash_dirty: bool = True` to GameState
2. Mark hash dirty when state changes:
   - When cards picked/discarded
   - When health changes
   - When equipped_weapon changes
   - When any state field changes
3. Calculate hash only when dirty
4. Return cached value otherwise
5. Update `_hash_state()` to check cache first

## Testing Considerations

- Verify hash matches current calculation
- Test hash uniqueness for different states
- Ensure cache invalidation works correctly
- Test hash collisions (should be rare)
- Compare performance improvement

## Expected Outcome

- **Performance**: 2-5% faster hash calculations
- **Code**: Slightly more complex (cache management)
- **Risk**: Low (straightforward caching)
