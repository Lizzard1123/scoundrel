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
    
    def __init__(self, scalar_input_dim: int):
        super(PolicySmallNet, self).__init__()
        
        # Input dimension: scalar_features + stack_sums (3 values)
        combined_input_dim = scalar_input_dim + 3
        
        self.scalar_fc = nn.Linear(combined_input_dim, SCALAR_ENCODER_OUT)
        self.shared_layer = nn.Linear(SCALAR_ENCODER_OUT, HIDDEN_DIM)
        self.action_head = nn.Linear(HIDDEN_DIM, ACTION_SPACE)

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

