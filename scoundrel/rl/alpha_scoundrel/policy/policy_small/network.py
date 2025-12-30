import torch
import torch.nn as nn
from scoundrel.rl.alpha_scoundrel.policy.policy_small.constants import (
    ACTION_SPACE,
)


class PolicySmallNet(nn.Module):
    """
    Policy network architecture with single FC layer.
    
    Architecture:
    - Single linear layer mapping combined features directly to action logits
    - Simplified from previous 3-layer architecture (now moved to policy large)
    
    Input:
    - scalar_features: Regular game state features [batch_size, scalar_input_dim]
    - stack_sums: Vector of scalars [batch_size, 3] for unknown health (potions), 
                  weapon, and monsters sums from dungeon stack
    """
    
    def __init__(
        self, 
        scalar_input_dim: int,
        action_space: int = None
    ):
        super(PolicySmallNet, self).__init__()
        
        action_space = action_space if action_space is not None else ACTION_SPACE
        
        # Input dimension: scalar_features + stack_sums (3 values) + total_stats (3 values)
        combined_input_dim = scalar_input_dim + 3 + 3
        
        # Single FC layer
        self.action_head = nn.Linear(combined_input_dim, action_space)

    def forward(self, scalar_data: torch.Tensor, stack_sums: torch.Tensor, total_stats: torch.Tensor):
        """
        Forward pass through the network.
        
        Args:
            scalar_data: [batch_size, scalar_input_dim] tensor of regular game state features
            stack_sums: [batch_size, 3] tensor of [potion_sum, weapon_sum, monster_sum] from unknown cards in dungeon stack
            total_stats: [batch_size, 3] tensor of [potion_sum, weapon_sum, monster_sum] from entire dungeon deck
            
        Returns:
            logits: [batch_size, ACTION_SPACE] tensor of action logits
        """
        # Concatenate scalar features with stack sums and total stats
        combined = torch.cat((scalar_data, stack_sums, total_stats), dim=1)
        
        # Pass through single FC layer
        return self.action_head(combined)

