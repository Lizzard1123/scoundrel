# How MCTS Picks Nodes: The UCB1 Formula

## Overview

MCTS picks nodes during the **Selection Phase** using the **UCB1 (Upper Confidence Bound 1)** formula. This balances two competing goals:
1. **Exploitation** - Pick actions that have worked well so far
2. **Exploration** - Try actions that haven't been explored much

## The UCB1 Formula

```python
UCB1(child) = exploitation + exploration
            = (child.value / child.visits) + C * sqrt(log(parent.visits) / child.visits)
            = average_reward + exploration_bonus
```

Where:
- `child.value` = Total reward accumulated from this child
- `child.visits` = Number of times this child was visited
- `parent.visits` = Number of times the parent was visited
- `C` = Exploration constant (default: √2 ≈ 1.414)

## How It Works in Code

```python
def best_child(self, exploration_constant: float) -> 'MCTSNode':
    best_score = float('-inf')
    best_child = None
    
    for child in self.children:
        # Special case: Always try unvisited children first
        if child.visits == 0:
            return child
        
        # UCB1 formula
        exploitation = child.value / child.visits           # Average reward
        exploration = exploration_constant * math.sqrt(
            math.log(self.visits) / child.visits
        )
        score = exploitation + exploration
        
        if score > best_score:
            best_score = score
            best_child = child
    
    return best_child
```

## Breaking Down Each Component

### 1. Exploitation Term: `child.value / child.visits`

This is the **average reward** from this action.

**Example:**
- Child A: 60 total value from 10 visits → 6.0 average
- Child B: 30 total value from 5 visits → 6.0 average

If we only used exploitation, we'd randomly pick between A and B.

### 2. Exploration Term: `C * sqrt(log(parent.visits) / child.visits)`

This **bonus** encourages trying actions that haven't been explored much.

**Key Properties:**
- Increases as parent gets more visits (parent.visits ↑)
- Decreases as child gets more visits (child.visits ↑)
- Never reaches zero (always some exploration)

**Example:** With C = 1.414, parent has 100 visits
- Child A (10 visits): exploration = 1.414 * sqrt(log(100)/10) ≈ 1.08
- Child B (5 visits): exploration = 1.414 * sqrt(log(100)/5) ≈ 1.53

Child B gets a bigger exploration bonus because it's been tried less!

### 3. Unvisited Children: Special Case

```python
if child.visits == 0:
    return child
```

Any child that has **never been visited** gets picked immediately. This ensures all actions are tried at least once before comparing them.

## Complete Example

Suppose we're at a node with 100 visits and 4 children:

| Child | Visits | Value | Avg Reward | Exploration | UCB1 Score | Picked? |
|-------|--------|-------|------------|-------------|------------|---------|
| A     | 0      | 0     | -          | ∞           | ∞          | ✓ Yes   |
| B     | 40     | 280   | 7.0        | 0.76        | 7.76       | -       |
| C     | 30     | 180   | 6.0        | 0.88        | 6.88       | -       |
| D     | 30     | 240   | 8.0        | 0.88        | 8.88       | -       |

**Step by step:**
1. Check child A: visits = 0 → **Pick immediately!**
2. (If A was already visited, we'd calculate all UCB1 scores)
3. Child D would win: highest UCB1 score (8.88)

## After Many Simulations

As simulations increase, the exploration term shrinks:

**After 1000 parent visits:**
- Child with 40 visits: exploration ≈ 0.65
- Child with 400 visits: exploration ≈ 0.20

The **best child eventually dominates** as its exploitation term outweighs others' exploration bonuses.

## The Exploration Constant (C)

Currently set to `MCTS_EXPLORATION_CONSTANT = 1.414` (√2).

### What it does:
- **Higher C** (e.g., 2.0): More exploration, tries diverse actions
- **Lower C** (e.g., 1.0): More exploitation, focuses on best known actions
- **C = √2**: Theoretically optimal balance (proven for certain conditions)

### Tuning C:

```python
# In constants.py
MCTS_EXPLORATION_CONSTANT = 1.414  # Default (balanced)

# Try these for different behaviors:
# 1.0 - Conservative, exploits more
# 2.0 - Adventurous, explores more
```

## Selection Phase Walkthrough

When MCTS selects a path through the tree:

```python
def _select(self, node, state):
    current_node = node
    current_state = state
    
    while current_node.is_fully_expanded() and current_node.children and not game_over:
        # Pick best child using UCB1
        current_node = current_node.best_child(self.exploration_constant)
        # Apply that action to the state
        current_state = apply_action(current_state, current_node.action)
    
    return current_node, current_state
```

At each step:
1. Calculate UCB1 for all children
2. Pick child with highest UCB1
3. Move to that child and update state
4. Repeat until we find a node to expand

## Final Action Selection

After all simulations are done, we pick the **most visited child** (NOT highest UCB1!):

```python
def most_visited_child(self):
    return max(self.children, key=lambda c: c.visits)
```

**Why visits instead of average reward?**
- Visits = how confident we are in this action
- An action with high average but few visits might just be lucky
- An action with many visits has been thoroughly tested
- This makes the final decision more robust

## Visualization Example

Imagine picking between 3 cards in Scoundrel:

```
Room: [5♥ Potion, 7♦ Weapon, 10♠ Monster]

After 100 simulations at this node:

Action: Take Potion (card 1)
├─ Visits: 25
├─ Total Value: 150
├─ Avg Reward: 6.0
├─ Exploration: 0.94
└─ UCB1: 6.94

Action: Take Weapon (card 2)
├─ Visits: 50
├─ Total Value: 400
├─ Avg Reward: 8.0
├─ Exploration: 0.66
└─ UCB1: 8.66  ← Highest! Pick this for next simulation

Action: Fight Monster (card 3)
├─ Visits: 25
├─ Total Value: 100
├─ Avg Reward: 4.0
├─ Exploration: 0.94
└─ UCB1: 4.94
```

**Next simulation:** MCTS picks "Take Weapon" because it has the highest UCB1 score (8.66).

**Final decision (after all sims):** "Take Weapon" because it has the most visits (50).

## Key Insights

1. **Early on:** Exploration dominates → tries all actions
2. **Middle phase:** Balanced → focuses on promising actions but still explores
3. **Late phase:** Exploitation dominates → converges on best action
4. **Unvisited nodes:** Always get tried first (infinite exploration bonus)
5. **Final choice:** Most visited = most confident, not necessarily highest reward

## Mathematical Guarantee

UCB1 has a **theoretical guarantee**: Given enough simulations, it will find the optimal action with probability approaching 1. The exploration term ensures we don't get stuck on suboptimal actions, while the exploitation term ensures we focus on good actions.

## Practical Performance

With your current settings:
```python
MCTS_NUM_SIMULATIONS = 2000
MCTS_EXPLORATION_CONSTANT = 1.414
```

- 2000 simulations gives excellent coverage
- Each action at root gets explored hundreds of times
- High confidence in final decision
- Good balance of exploration/exploitation
- Should make near-optimal decisions

## Further Reading

The UCB1 formula comes from the **Multi-Armed Bandit** problem in reinforcement learning. MCTS applies this to tree search by treating each node as a multi-armed bandit problem where "arms" are the possible actions (children).



