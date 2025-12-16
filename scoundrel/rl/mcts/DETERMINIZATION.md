# MCTS Determinization for Hidden Information

## The Problem: Partial Observability

Scoundrel is a game with **hidden information**. The dungeon deck has cards in a specific order, but the player doesn't know that order initially.

### What the Agent Knows

1. **Room Cards** - Always visible (4 cards currently in play)
2. **Known Dungeon Cards** - Cards that have been seen and avoided
3. **Unknown Dungeon Cards** - Cards not yet revealed

### How "Avoid" Works

When you avoid a room:
1. All 4 room cards go to the **bottom** of the dungeon deck
2. The agent now **knows** those 4 cards and their positions
3. `number_avoided` is incremented
4. The cards at indices `>= number_avoided * 4` are **known**
5. The cards at indices `< number_avoided * 4` are **unknown**

### Example

```
Initial dungeon: [?, ?, ?, ?, ?, ?, ..., ?]  (52 cards, all unknown)
number_avoided = 0

After avoiding once with cards [A, B, C, D]:
Dungeon: [?, ?, ?, ?, ?, ?, ..., A, B, C, D]
         └─────── unknown ─────┘  └─ known ─┘
number_avoided = 1
Unknown cards: indices 0-3 (4 cards)
Known cards: indices 4-51 (48 cards at end)

After avoiding twice with cards [E, F, G, H]:
Dungeon: [?, ?, ?, ?, ?, ?, ..., A, B, C, D, E, F, G, H]
         └─────── unknown ─────┘  └───── known ─────┘
number_avoided = 2
Unknown cards: indices 0-7 (8 cards)
Known cards: indices 8-51 (44 cards at end)
```

## The Solution: Determinization

MCTS needs to handle this hidden information by using **determinization**:

1. For each simulation, sample a possible ordering of unknown cards
2. Keep known cards in their correct positions
3. Play out the simulation with this sampled ordering
4. Different simulations use different random orderings
5. Aggregate results across all simulations

### Implementation

```python
def _determinize_state(self, game_state: GameState) -> GameState:
    """
    Shuffles unknown cards while preserving known card positions.
    """
    determinized_state = copy.deepcopy(game_state)
    
    if determinized_state.number_avoided > 0:
        unknown_count = determinized_state.number_avoided * 4
        if unknown_count < len(determinized_state.dungeon):
            # Split into unknown and known portions
            unknown_cards = determinized_state.dungeon[:unknown_count]
            known_cards = determinized_state.dungeon[unknown_count:]
            
            # Shuffle only unknown cards
            random.shuffle(unknown_cards)
            
            # Reconstruct with shuffled unknown + preserved known
            determinized_state.dungeon = unknown_cards + known_cards
    
    return determinized_state
```

### Usage in MCTS

Each simulation starts with a fresh determinization:

```python
for _ in range(self.num_simulations):
    # Sample one possible world consistent with known information
    simulation_state = self._determinize_state(game_state)
    
    # Run simulation with this determinization
    node, simulation_state = self._select(root, simulation_state)
    node, simulation_state = self._expand(node, simulation_state)
    reward = self._simulate(simulation_state)
    self._backpropagate(node, reward)
```

## Why This Matters

### Without Determinization (Cheating)
```python
# Agent sees exact dungeon order: [2♣, 5♠, K♦, A♥, ...]
# Makes decisions based on perfect information
# Unrealistic and won't work in actual play
```

### With Determinization (Realistic)
```python
# Simulation 1: [5♠, 2♣, K♦, A♥, ...] (unknown shuffled)
# Simulation 2: [K♦, 5♠, 2♣, A♥, ...] (different shuffle)
# Simulation 3: [2♣, K♦, 5♠, A♥, ...] (different shuffle)
# Agent makes decisions robust to uncertainty
# Works with actual partial information
```

## Comparison with Transformer

The transformer handles this in the encoding phase (translator.py):

```python
for i, elm in enumerate(game_state.dungeon):
    if i >= game_state.number_avoided * 4:
        stack_ids.append(self._card_to_id(elm))  # Encode known cards
    else:
        stack_ids.append(0)  # Encode unknown cards as padding
```

MCTS handles it in the simulation phase:
- Samples possible orderings
- Evaluates each through tree search
- Aggregates results to make robust decisions

## Benefits

1. **Realistic Play** - Doesn't rely on information the agent shouldn't have
2. **Robust Decisions** - Considers multiple possible futures
3. **Exploits Known Information** - Uses knowledge from avoided rooms
4. **Fair Comparison** - Can be compared with transformer which also hides unknown cards

## Technical Note: Information Set MCTS

This approach is known as "Information Set MCTS" (ISMCTS):
- Each simulation samples a determinization (possible world)
- Tree nodes represent information sets (what the agent knows)
- Actions are evaluated across multiple determinizations
- Converges to optimal play under perfect information game equivalence

## Testing

To verify determinization is working:

```python
# Create state with some avoided rooms
state.number_avoided = 2  # 8 cards should be unknown

# Run multiple determinizations
det1 = agent._determinize_state(state)
det2 = agent._determinize_state(state)

# First 8 cards should be different (shuffled)
assert det1.dungeon[:8] != det2.dungeon[:8]

# Known cards should be same (preserved)
assert det1.dungeon[8:] == det2.dungeon[8:]
```

## Performance Impact

Determinization adds minimal overhead:
- One shuffle operation per simulation
- O(n) where n = number of unknown cards
- Typically 0-40 cards (0-10 avoids)
- Negligible compared to simulation cost

The benefit far outweighs the cost:
- Prevents cheating
- Enables realistic evaluation
- Improves decision quality
- Makes MCTS comparable to other methods



