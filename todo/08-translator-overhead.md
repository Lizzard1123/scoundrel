# Issue #8: Translator Overhead (LOW)

## Priority: LOW
**Estimated Impact: 1-3% speedup potential**

## Problem Description

`self.translator.decode_action(action)` is called frequently:

- Simple enum mapping: `Action(action_idx)` for most cases
- Function call overhead adds up
- Called once per action application (millions of times per move)

## Root Cause

- Function call overhead for simple operation
- Enum mapping is straightforward
- No caching of mappings

## Current Code Locations

- `scoundrel/rl/translator.py:103-107` - `decode_action()` method
- `scoundrel/rl/mcts/mcts_agent.py:564` - Called in `_select()`
- `scoundrel/rl/mcts/mcts_agent.py:586` - Called in `_expand()`
- `scoundrel/rl/mcts/mcts_agent.py:637` - Called in `_simulate()` (after Issue #1 fix)

## Impact Analysis

- **Frequency**: 
  - Once per action application
  - Millions of times per move
- **Cost**: 
  - Function call overhead
  - Enum lookup
- **Total overhead**: Small but measurable

## Proposed Solutions

### Option 1: Cache Action Enum Mappings (RECOMMENDED)
- Pre-compute action enum mappings
- Use lookup table instead of function call
- Simple optimization

### Option 2: Inline Mapping
- Inline the mapping logic where used
- Avoid function call overhead
- Less clean but faster

### Option 3: Keep Current
- Overhead may be negligible
- Function call is clear and readable
- May not be worth optimizing

## Recommended Approach

**Option 1** (Cache Action Enum Mappings) - Best balance:
- Simple to implement
- Good performance improvement
- Maintains code clarity

## Implementation Notes

1. Create action enum cache in MCTSAgent:
   ```python
   def __init__(self, ...):
       # ... existing code ...
       self._action_enum_cache = {
           0: Action.USE_1,
           1: Action.USE_2,
           2: Action.USE_3,
           3: Action.USE_4,
           4: Action.AVOID
       }
   ```

2. Use cache instead of translator:
   ```python
   action_enum = self._action_enum_cache[action]
   ```

3. Or inline: `action_enum = list(Action)[action]` if action < 4 else Action.AVOID

## Testing Considerations

- Verify action mapping is correct
- Test all action types
- Compare performance improvement
- Ensure no regressions

## Expected Outcome

- **Performance**: 1-3% faster action decoding
- **Code**: Slightly more complex (cache management)
- **Risk**: Low (simple optimization)
