"""
AlphaGo MCTS Node implementation for Scoundrel.
Enhanced MCTS node that stores policy priors from neural networks.
"""
import math
import torch
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class AlphaGoNode:
    """
    A node in the AlphaGo-style MCTS search tree.
    
    Key enhancement over vanilla MCTS: stores policy priors P(s,a) from neural network
    for use in PUCT (Predictor + UCB applied to Trees) formula.
    
    Attributes:
        state_hash: Hash representing the game state
        parent: Parent node in the tree
        action: Action taken to reach this node from parent
        children: List of child nodes
        visits: Number of times this node has been visited
        value: Total value accumulated from simulations
        untried_actions: Actions that haven't been explored yet
        prior_probs: Policy priors P(s,·) from neural network for PUCT
        is_game_over: Whether this node represents a terminal game state
    """
    state_hash: int
    parent: Optional['AlphaGoNode'] = None
    action: Optional[int] = None
    children: List['AlphaGoNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: List[int] = field(default_factory=list)
    prior_probs: Optional[torch.Tensor] = None  # NEW: P(s,·) from policy network
    is_game_over: bool = False
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried."""
        return len(self.untried_actions) == 0
    
    def best_child_puct(self, c_puct: float) -> 'AlphaGoNode':
        """
        Select the best child using PUCT formula (policy-enhanced UCB).
        
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        where:
        - Q(s,a) = mean action value (exploitation)
        - P(s,a) = policy prior from neural network
        - N(s,a) = visit count for action a
        - c_puct = exploration constant (analogous to UCB1's c)
        
        Key difference from vanilla UCB1:
        - Uses policy priors P(s,a) instead of uniform exploration
        - Explores promising actions (high P) more than unpromising ones
        
        Args:
            c_puct: PUCT exploration constant
            
        Returns:
            Child node with highest PUCT value
        """
        best_score = float('-inf')
        best_child = None
        
        # If no priors available, fall back to uniform exploration (vanilla UCB1)
        if self.prior_probs is None:
            return self.best_child_ucb1(c_puct)
        
        sqrt_parent_visits = math.sqrt(self.visits)
        
        for child in self.children:
            # Handle unvisited nodes (give them highest priority)
            if child.visits == 0:
                return child
            
            # Exploitation term: Q(s,a) = mean value
            q_value = child.value / child.visits
            
            # Exploration term: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            # P(s,a) comes from policy network priors
            prior_prob = self.prior_probs[child.action].item()
            exploration = c_puct * prior_prob * sqrt_parent_visits / (1 + child.visits)
            
            # PUCT = Q + U
            puct_score = q_value + exploration
            
            if puct_score > best_score:
                best_score = puct_score
                best_child = child
        
        return best_child
    
    def best_child_ucb1(self, exploration_constant: float) -> 'AlphaGoNode':
        """
        Fallback to vanilla UCB1 if no policy priors available.
        
        UCB1 = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
        
        Args:
            exploration_constant: Balance between exploitation and exploration
            
        Returns:
            Child node with highest UCB1 value
        """
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                return child
            
            exploitation = child.value / child.visits
            exploration = exploration_constant * math.sqrt(
                math.log(self.visits) / child.visits
            )
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def set_prior_probs(self, probs: torch.Tensor):
        """
        Store policy prior probabilities for valid actions.
        
        Args:
            probs: Tensor of action probabilities [action_space] from policy network
        """
        self.prior_probs = probs
    
    def most_visited_child(self) -> 'AlphaGoNode':
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
        return (f"AlphaGoNode(action={self.action}, visits={self.visits}, "
                f"value={self.value:.2f}, children={len(self.children)})")

