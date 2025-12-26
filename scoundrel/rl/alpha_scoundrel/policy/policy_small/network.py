import torch
import torch.nn as nn
import torch.nn.functional as F
from scoundrel.rl.alpha_scoundrel.policy.policy_small.constants import (
    ACTION_SPACE,
    SCALAR_ENCODER_OUT,
    HIDDEN_DIM,
)


class PolicySmallNet(nn.Module):
    """
    Policy network architecture with single MLP.
    
    Architecture:
    - MLP for scalar features (current room & status + dungeon stack sums)
    - Single actor head (no critic)
    
    Input:
    - scalar_features: Regular game state features [batch_size, scalar_input_dim]
    - stack_sums: Vector of scalars [batch_size, 3] for unknown health (potions), 
                  weapon, and monsters sums from dungeon stack
    """
    
    def __init__(
        self, 
        scalar_input_dim: int,
        scalar_encoder_out: int = None,
        hidden_dim: int = None,
        action_space: int = None
    ):
        super(PolicySmallNet, self).__init__()
        
        # Use provided constants or fall back to defaults
        scalar_encoder_out = scalar_encoder_out if scalar_encoder_out is not None else SCALAR_ENCODER_OUT
        hidden_dim = hidden_dim if hidden_dim is not None else HIDDEN_DIM
        action_space = action_space if action_space is not None else ACTION_SPACE
        
        # Input dimension: scalar_features + stack_sums (3 values)
        combined_input_dim = scalar_input_dim + 3
        
        self.scalar_fc = nn.Linear(combined_input_dim, scalar_encoder_out)
        self.shared_layer = nn.Linear(scalar_encoder_out, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_space)

    def forward(self, scalar_data: torch.Tensor, stack_sums: torch.Tensor):
        """
        Forward pass through the network.
        
        Args:
            scalar_data: [batch_size, scalar_input_dim] tensor of regular game state features
            stack_sums: [batch_size, 3] tensor of [potion_sum, weapon_sum, monster_sum] from dungeon stack
            
        Returns:
            logits: [batch_size, ACTION_SPACE] tensor of action logits
        """
        # Concatenate scalar features with stack sums
        combined = torch.cat((scalar_data, stack_sums), dim=1)
        
        # Pass through MLP
        x = F.relu(self.scalar_fc(combined))
        x = F.relu(self.shared_layer(x))
        return self.action_head(x)

