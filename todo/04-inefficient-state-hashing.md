# Issue #4: Inefficient State Hashing

## Priority: HIGH
**Estimated Impact: 5-10% speedup potential**

## Problem Description

`_hash_state()` creates state hashes using string concatenation:

- Iterates through room cards creating strings: `f"{c.suit.value}{c.value}"`
- Iterates through first 5 dungeon cards
- Multiple string concatenations and formatting operations
- Called for every node creation

## Root Cause

- String operations are slower than integer operations
- String concatenation creates new string objects
- Hash is used for node identification but string comparison is slower

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:304-310` - `_hash_state()` method
- `scoundrel/rl/mcts/mcts_agent.py:293` - Called in `_create_node()`

## Impact Analysis

- **Frequency**: Called once per node creation
- **Cost**: 
  - String formatting for each card
  - String concatenation operations
  - String comparison during node lookup (if used)
- **Total overhead**: Moderate, but adds up with many nodes

## Proposed Solutions

### Option 1: Integer-Based Hashing
- Use Python's `hash()` on tuple of state values
- Convert cards to integers: `card.suit.value * 100 + card.value`
- Create tuple: `(health, weapon_value, room_tuple, dungeon_tuple)`
- Use `hash(tuple)` for fast integer hash

### Option 2: Custom Hash Function
- Combine state values using bit operations
- Shift and XOR operations for fast hashing
- Produces integer hash directly

### Option 3: Use Built-in Hash
- Make GameState hashable (implement `__hash__`)
- Use Python's built-in hashing
- Requires making GameState immutable or using frozenset

### Option 4: Skip Hashing Entirely
- If hash is only used for debugging/display, remove it
- Use node reference comparison instead
- Only hash when needed for transposition table

## Recommended Approach

**Option 1** (Integer-Based Hashing) - Best balance:
- Fast integer operations
- Can use Python's built-in `hash()`
- Easy to implement
- Still readable for debugging

## Implementation Notes

- Convert Card to integer: `suit_enum_value * 100 + card_value`
- Create tuple of key state features
- Use `hash()` on tuple for consistent hashing
- Consider caching hash if state doesn't change

## Testing Considerations

- Verify hash uniqueness for different states
- Test hash collisions (should be rare)
- Ensure hash is deterministic
- Compare performance improvement
