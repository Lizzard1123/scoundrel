import torch
import torch.nn as nn
import torch.nn.functional as F
from scoundrel.rl.alpha_scoundrel.policy.policy_large.constants import (
    ACTION_SPACE,
    CARD_EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_CARDS,
    SCALAR_ENCODER_OUT,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NLAYERS,
)


class PolicyLargeNet(nn.Module):
    """
    Policy network architecture identical to ScoundrelNet but with only actor head.
    
    Architecture:
    - Transformer encoder for sequence data (dungeon stack)
    - MLP for scalar features (current room & status)
    - Single actor head (no critic)
    """
    
    def __init__(self, scalar_input_dim):
        super(PolicyLargeNet, self).__init__()

        self.embedding = nn.Embedding(NUM_CARDS, CARD_EMBEDDING_DIM, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CARD_EMBEDDING_DIM,
            nhead=TRANSFORMER_NHEAD,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_NLAYERS)

        self.scalar_fc = nn.Linear(scalar_input_dim, SCALAR_ENCODER_OUT)

        combined_dim = CARD_EMBEDDING_DIM + SCALAR_ENCODER_OUT

        self.shared_layer = nn.Linear(combined_dim, HIDDEN_DIM)
        self.action_head = nn.Linear(HIDDEN_DIM, ACTION_SPACE)

    def forward(self, scalar_data, sequence_data):
        """
        Forward pass through the network.
        
        Args:
            scalar_data: [batch_size, scalar_input_dim] tensor
            sequence_data: [batch_size, seq_len] tensor of card IDs
                         (0 = padding or unknown card)
            
        Returns:
            logits: [batch_size, ACTION_SPACE] tensor of action logits
        """
        embedded = self.embedding(sequence_data)

        # Create attention mask: True for padding/unknown cards (0), False for valid cards
        # PyTorch transformer uses src_key_padding_mask where True = ignore
        padding_mask = (sequence_data == 0)

        # MPS doesn't support src_key_padding_mask, so skip it on MPS devices
        # We'll still handle padding correctly via masked pooling below
        device = embedded.device
        if device.type == 'mps':
            # On MPS, don't pass the padding mask to transformer
            trans_out = self.transformer(embedded)
        else:
            # On CPU/CUDA, use padding mask for efficiency
            trans_out = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Use masked mean pooling: only average over valid (non-padding/unknown) cards
        # Create mask for valid positions (inverse of padding_mask)
        valid_mask = ~padding_mask  # [batch_size, seq_len]
        
        # Set padding positions to 0 for masked sum
        masked_trans_out = trans_out * valid_mask.unsqueeze(-1).float()
        
        # Sum over sequence dimension, then divide by count of valid tokens
        valid_counts = valid_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
        # Avoid division by zero (shouldn't happen, but safety check)
        valid_counts = torch.clamp(valid_counts, min=1.0)
        seq_features = masked_trans_out.sum(dim=1) / valid_counts  # [batch_size, CARD_EMBEDDING_DIM]

        scal_features = F.relu(self.scalar_fc(scalar_data))

        combined = torch.cat((seq_features, scal_features), dim=1)
        x = F.relu(self.shared_layer(combined))
        return self.action_head(x)

