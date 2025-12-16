# Understanding MCTS Values: Win vs Loss

## The Confusion

When looking at MCTS values like **0.76** or **0.85**, they seem "high" - but they actually represent **losing outcomes**!

## The Key Insight

**The survival threshold is ~0.867**

- Values **> 0.867** = You survived (score > 0)
- Values **< 0.867** = You died (score ≤ 0)

## Why This Happens

Our normalization maps the full score range to [0, 1]:

```python
normalized = (score + 188) / 218
```

This means:
- Score -188 (worst death) → 0.000
- Score 0 (die at end) → **0.862**
- Score 1 (barely survive) → **0.867**
- Score 30 (perfect) → 1.000

**The middle value (0.5) is NOT the win/loss threshold!**

## Value Interpretation Table

| Normalized | Raw Score | Outcome | Meaning |
|------------|-----------|---------|---------|
| 0.00-0.40 | -188 to -100 | Loss | Died early, many monsters left |
| 0.40-0.70 | -100 to -30 | Loss | Died midgame |
| 0.70-0.86 | -30 to 0 | Loss | Died late game |
| **0.867** | **1** | **Threshold** | **Barely survived** |
| 0.87-0.95 | 1 to 20 | Win | Survived with low-good HP |
| 0.95-1.00 | 20 to 30 | Win | Excellent outcome |

## Your Example: Two Losing Options

When you observed two options both around **0.76**:

```
Option A: 0.77 → score ≈ -20 (die with 20 monster value left)
Option B: 0.75 → score ≈ -25 (die with 25 monster value left)
```

**Both are losses!** But:
- MCTS picks 0.77 because it's "less bad"
- It's choosing to die with fewer monsters remaining
- But it still dies

This is correct MCTS behavior: when all options are bad, pick the least bad one.

## Why MCTS Shows High Values for Losing Positions

1. **Late-game losses look "better"** than early-game losses
2. **Dying at 0 HP is "better"** (0.862) than dying at -50 HP (0.633)
3. **The full range includes very bad outcomes** (-188), so moderate losses (0 to -50) appear "middle-ish"

## Updated Viewer Display

The viewer now shows win/loss indicators:

```
Next: [use 2] | 1:0.76/450✗ [2:0.85/892✗] 3:0.72/380✗ 4:0.69/278✗
```

- **✓** (green) = Value > 0.867 (expected to survive)
- **✗** (dim/yellow) = Value < 0.867 (expected to die)
- **[brackets]** = Selected action

## Common Scenarios

### All Good Options
```
1:0.94/500✓ [2:0.96/800✓] 3:0.91/400✓ 4:0.89/300✓
```
All options lead to survival, pick best one (0.96)

### Mixed Options  
```
1:0.72/500✗ [2:0.92/800✓] 3:0.68/400✗ 4:0.65/300✗
```
Only option 2 survives, MCTS correctly picks it

### All Bad Options (Your Case)
```
[1:0.77/500✗] 2:0.75/450✗ 3:0.73/350✗ 4:0.70/300✗
```
All options die, pick least-bad (die at score -20 vs -30)

### Near Threshold
```
1:0.86/500✗ [2:0.87/800✓] 3:0.85/400✗
```
Close call! Option 2 barely survives (score ~1-2 HP)

## Why This Normalization Makes Sense

### For Neural Networks (Transformer)
- Needs [0, 1] range for stable training
- Can learn gradations of "how bad" a loss is
- Advantage calculation works better

### For MCTS
- UCB1 formula assumes [0, 1] values
- Can distinguish between different qualities of losses
- Helps pick "least bad" option when trapped

## Alternative: Win/Loss Only

You might think: "Why not just use 0 for any loss, 1 for any win?"

**Problem:** Loses information about:
- How close you were to winning
- Whether to pick "die at -10" vs "die at -100"
- Quality of wins (20 HP vs 1 HP)

Our current normalization preserves this information while keeping values in [0, 1].

## Key Takeaway

**Don't confuse normalized values with win probability!**

- 0.76 doesn't mean "76% chance to win"
- It means "average final score of -22"
- Which is a loss, but not the worst possible loss

The **win/loss threshold is ~0.867**, not 0.5!



