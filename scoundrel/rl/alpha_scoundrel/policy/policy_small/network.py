import torch
import torch.nn as nn
from scoundrel.rl.alpha_scoundrel.policy.policy_small.constants import (
    ACTION_SPACE,
)


class PolicySmallNet(nn.Module):
    """
    Policy network architecture with single FC layer + regularization.

    Architecture:
    - Single linear layer with dropout regularization
    - Advanced training features: gradient clipping, focal loss, temperature sharpening
    - Enhanced from basic architecture with modern training techniques

    Input:
    - scalar_features: Regular game state features [batch_size, scalar_input_dim]
    - stack_sums: Vector of scalars [batch_size, 3] for unknown health (potions),
                  weapon, and monsters sums from dungeon stack
    """

    def __init__(
        self,
        scalar_input_dim: int,
        action_space: int = None,
        dropout_rate: float = 0.1
    ):
        super(PolicySmallNet, self).__init__()

        action_space = action_space if action_space is not None else ACTION_SPACE

        # Input dimension: scalar_features + stack_sums (3 values)
        combined_input_dim = scalar_input_dim + 3

        # Single FC layer with dropout
        self.action_head = nn.Linear(combined_input_dim, action_space)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, scalar_data: torch.Tensor, stack_sums: torch.Tensor):
        """
        Forward pass through the network with regularization.

        Args:
            scalar_data: [batch_size, scalar_input_dim] tensor of regular game state features
            stack_sums: [batch_size, 3] tensor of [potion_sum, weapon_sum, monster_sum] from dungeon stack

        Returns:
            logits: [batch_size, ACTION_SPACE] tensor of action logits
        """
        # Concatenate scalar features with stack sums
        combined = torch.cat((scalar_data, stack_sums), dim=1)

        # Apply dropout for regularization
        combined = self.dropout(combined)

        # Pass through single FC layer
        return self.action_head(combined)

