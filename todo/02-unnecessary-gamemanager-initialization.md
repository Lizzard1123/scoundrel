# Issue #2: Unnecessary GameManager Initialization Overhead

## Priority: CRITICAL
**Estimated Impact: 10-20% speedup potential**

## Problem Description

`_create_engine_from_state()` creates a full `GameManager()` instance which performs unnecessary initialization:

1. Generates random seed (lines 22-23) - immediately discarded
2. Creates `TerminalUI` instance (line 28) - **never used** in MCTS
3. Calls `setup_game()` which creates full deck (line 32) - **immediately overwritten**
4. Then overwrites `engine.state` with copied state (line 493)

## Root Cause

- `GameManager.__init__()` is designed for interactive gameplay, not MCTS
- MCTS only needs the `execute_turn()` functionality, not UI or deck setup
- Full initialization happens before state is overwritten

## Current Code Locations

- `scoundrel/rl/mcts/mcts_agent.py:490-494` - `_create_engine_from_state()`
- `scoundrel/game/game_manager.py:12-29` - `GameManager.__init__()`

## Impact Analysis

- **Frequency**: Called multiple times per simulation:
  - Once in `_select()` (line 329)
  - Once in `_expand()` (line 348)  
  - Once in `_simulate()` (line 364)
  - Plus multiple times during simulation rollout
- **Cost**: 
  - TerminalUI creation (unnecessary)
  - Deck creation via `Deck.create_deck()` (~44 cards)
  - Random seed generation
- **Total overhead**: Significant waste, especially in simulation phase

## Proposed Solutions

### Option 1: Lightweight MCTS GameManager
Create a minimal `MCTSGameManager` class that:
- Only contains state and `execute_turn()` logic
- No UI initialization
- No deck setup
- Can be initialized directly with a GameState

### Option 2: Factory Method
Add `GameManager.from_state(state)` class method:
- Bypasses normal initialization
- Sets state directly
- Skips UI and deck creation

### Option 3: Lazy Initialization
Modify `GameManager` to support lazy initialization:
- Add optional parameter to skip UI/deck setup
- Only initialize what's needed

### Option 4: Extract Game Logic
Extract game logic from `GameManager` into a pure function or lightweight class:
- `execute_turn()` becomes stateless function: `(state, action) -> new_state`
- No object creation needed

## Recommended Approach

**Option 1** (Lightweight MCTS GameManager) - Clean separation:
- Keeps MCTS code isolated
- No risk of breaking interactive gameplay
- Easy to optimize further

## Implementation Notes

- Need to extract `execute_turn()` logic into reusable method
- Must handle all action types correctly
- Ensure state mutations are correct

## Testing Considerations

- Verify all action types work correctly
- Test that simulations produce same results
- Ensure no side effects from missing UI/deck
