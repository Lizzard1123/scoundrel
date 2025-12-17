"""
MCTS Node implementation for Scoundrel.
Each node represents a game state in the search tree.
"""
import math
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """
    A node in the MCTS search tree.
    
    Attributes:
        state_hash: Hash representing the game state
        parent: Parent node in the tree
        action: Action taken to reach this node from parent
        children: List of child nodes
        visits: Number of times this node has been visited
        value: Total value accumulated from simulations
        untried_actions: Actions that haven't been explored yet
        is_game_over: Whether this node represents a terminal game state
    """
    state_hash: int
    parent: Optional['MCTSNode'] = None
    action: Optional[int] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[int] = field(default_factory=list)
    is_game_over: bool = False
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried."""
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_constant: float) -> 'MCTSNode':
        """
        Select the best child using UCB1 formula.
        
        Args:
            exploration_constant: Balance between exploitation and exploration
            
        Returns:
            Child node with highest UCB1 value
        """
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                # Prioritize unvisited children
                return child
            
            # UCB1 formula: exploitation + exploration
            exploitation = child.value / child.visits
            exploration = exploration_constant * math.sqrt(
                math.log(self.visits) / child.visits
            )
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def most_visited_child(self) -> 'MCTSNode':
        """Return the child with the most visits (used for final action selection)."""
        return max(self.children, key=lambda c: c.visits)
    
    def update(self, reward: float):
        """
        Update this node with the result of a simulation.
        
        Args:
            reward: Reward from the simulation
        """
        self.visits += 1
        self.value += reward
    
    def __repr__(self):
        return (f"MCTSNode(action={self.action}, visits={self.visits}, "
                f"value={self.value:.2f}, children={len(self.children)})")

