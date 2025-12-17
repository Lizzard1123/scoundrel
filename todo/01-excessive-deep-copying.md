# Issue #1: Excessive Deep Copying (CRITICAL)

## Priority: CRITICAL
**Estimated Impact: 50-80% speedup potential**

## Problem Description

The MCTS implementation performs `copy.deepcopy()` operations excessively:

1. **In `_determinize_state()`** (line 472): Deep copies entire GameState before determinization
2. **In `_create_engine_from_state()`** (line 493): Deep copies GameState again when creating engine

With 10,000 simulations per move, this results in **20,000 deep copy operations** per move.

## Root Cause

- GameState contains multiple lists of Card objects:
  - `dungeon`: ~40 Card objects
  - `room`: 4 Card objects  
  - `discard`: Variable number of Card objects
  - `weapon_monsters`: Variable number of Card objects
- Each Card is a dataclass that gets fully copied
- Deep copying is O(n) where n = total number of objects in the state tree

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:472` - `_determinize_state()`
- `scoundrel/rl/mcts/mcts_agent.py:493` - `_create_engine_from_state()`

## Impact Analysis

- **Frequency**: Called once per simulation (10,000+ times per move)
- **Cost**: Each deep copy traverses ~50+ Card objects plus lists
- **Total overhead**: Massive - likely the #1 performance bottleneck

## Proposed Solutions

### Option 1: Shallow Copy with Selective Deep Copy
- Shallow copy GameState (fast)
- Only deep copy mutable collections (dungeon, room, discard, weapon_monsters)
- Card objects are immutable (dataclass with value/suit), so shallow copy may suffice

### Option 2: Custom Copy Method
- Implement `GameState.copy()` method that does optimized copying
- Only copy what's actually needed for determinization/simulation
- Use list slicing instead of deep copy where possible

### Option 3: State Pooling
- Maintain a pool of GameState objects
- Reuse states instead of copying
- Reset state fields instead of creating new objects

### Option 4: Immutable State Pattern
- Make GameState immutable
- Return new states from actions instead of mutating
- Eliminates need for copying (but requires refactoring)

## Recommended Approach

**Option 1** (Shallow Copy with Selective Deep Copy) - Best balance of performance and safety:
- Cards are effectively immutable (value/suit don't change)
- Only need to copy list references, not Card objects themselves
- Can use `list.copy()` or slicing for lists

## Testing Considerations

- Verify determinization still works correctly
- Ensure simulations don't interfere with each other
- Test that parallel workers don't share state accidentally
