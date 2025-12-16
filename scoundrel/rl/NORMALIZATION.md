# Score Normalization in Scoundrel RL

## Overview

Both MCTS and Transformer agents normalize game scores to the [0, 1] range for reward signals.

## Actual Deck Composition

Scoundrel uses a **44-card deck**:
- **Hearts (Potions)**: 2-10 (9 cards) → Max potion = **10**
- **Diamonds (Weapons)**: 2-10 (9 cards)  
- **Clubs (Monsters)**: 2-14 (13 cards)
- **Spades (Monsters)**: 2-14 (13 cards)
- **Total**: 44 cards

Face cards (J, Q, K, A) are **excluded** from red suits (Hearts/Diamonds).

**Game Start:**
- 4 cards dealt face-up to room
- 40 cards remain in dungeon deck (unknown at start)

## Score Bounds

### Maximum Score: 30
- Finish with 20 HP (max health)
- With a 10 potion (max value) as the last potion used
- Score = 20 + 10 = **30**

### Minimum Score: -188
- Die at exactly 0 HP (takes 20 damage to die from full health)
- Maximum monsters remaining: 208 - 20 = 188
- Score = 0 - 188 = **-188**

### Range: 218
- From -188 to 30
- Total range = 30 - (-188) = 218

## Normalization Formula

```python
normalized_reward = (score - min_score) / (max_score - min_score)
                  = (score - (-188)) / (30 - (-188))
                  = (score + 188) / 218
```

### Examples

| Raw Score | Calculation | Normalized | Meaning |
|-----------|-------------|------------|---------|
| -188 | (-188 + 188) / 218 | 0.000 | Worst possible (died at 0 HP, max monsters left) |
| -100 | (-100 + 188) / 218 | 0.404 | Died with negative HP midgame |
| 0 | (0 + 188) / 218 | 0.862 | Died at end, exactly 0 HP |
| 10 | (10 + 188) / 218 | 0.908 | Survived with 10 HP |
| 20 | (20 + 188) / 218 | 0.954 | Perfect health, no bonus |
| 30 | (30 + 188) / 218 | 1.000 | Perfect game (20 HP + 10 potion) |

## Why This Matters

### Corrected Misconceptions

**Previous Incorrect Assumptions:**
- ❌ 52-card deck (actually 44)
- ❌ Max potion = 14/Ace (actually 10)
- ❌ Max score = 34 (actually 30)
- ❌ Min score = -208 (actually -188, can't have all monsters remaining if you died)
- ❌ Dungeon size = 44 (actually 40, since 4 dealt to room at start)

**Corrected Values:**
- ✓ 44-card deck total
- ✓ Max potion = 10
- ✓ Max score = 30
- ✓ Min score = -188 (takes 20 damage to die, leaving 188 monster value)
- ✓ Dungeon starts with 40 cards (4 in room)
- ✓ Range = 218

## Implementation

### MCTS Agent
```python
def _normalize_reward(self, score: int) -> float:
    min_score = -188
    max_score = 30
    return (score - min_score) / (max_score - min_score)
```

### Transformer PPO
```python
# During training
reward = ((next_state.score + 188) / 218) if done else 0

# Convert back for logging
raw_score = normalized_reward * 218 - 188
```

## Card Encoding Updates

The translator now correctly maps the 44-card deck:

```python
suit_offsets = {
    Suit.CLUBS: 0,      # IDs 1-13 (cards 2-14)
    Suit.SPADES: 13,    # IDs 14-26 (cards 2-14)
    Suit.DIAMONDS: 26,  # IDs 27-35 (cards 2-10)
    Suit.HEARTS: 35     # IDs 36-44 (cards 2-10)
}
```

- Embedding size: 45 (0=padding, 1-44=cards)
- Sequence length: 40 (dungeon size at game start)

## Impact on Models

### MCTS
- Now correctly handles 44-card simulations
- Proper bounds for normalization [0, 1]
- Determinization shuffles correct number of cards

### Transformer PPO
- Embedding dimension: 45 (0=pad, 1-44=cards)
- Sequence length: 40 (dungeon at start)
- Rewards properly normalized with correct bounds
- Network will need retraining with corrected normalization (architecture already correct!)

## Typical Performance Ranges

| Normalized Value | Raw Score Equiv | Performance |
|------------------|-----------------|-------------|
| 0.00 - 0.30 | -188 to -123 | Very poor (early death with many monsters left) |
| 0.30 - 0.60 | -123 to -53 | Poor (mid-game death) |
| 0.60 - 0.80 | -53 to 4 | Below average (barely survived or late death) |
| 0.80 - 0.90 | 4 to 15 | Average (low-medium HP survival) |
| 0.90 - 0.95 | 15 to 24 | Good (high HP survival) |
| 0.95 - 1.00 | 24 to 30 | Excellent (near perfect) |

## Viewer Display

When you see in the MCTS viewer:
```
Next: [use 2] | 1:0.89/450 [2:0.93/892] 3:0.87/380 4:0.85/278
```

Interpretation with corrected normalization:
- Card 1: 0.89 → ~6 HP expected outcome
- Card 2: 0.93 → ~15 HP expected outcome (best!)
- Card 3: 0.87 → ~1 HP expected outcome
- Card 4: 0.85 → -4 HP expected outcome (likely die)

## Breaking Change

**Important:** Existing trained transformer models may need retraining:
- Architecture was already correct (45 embeddings, 40 sequence)
- But normalization was wrong (used 238 range instead of 218)
- **Retrain for best performance** with corrected reward scaling

