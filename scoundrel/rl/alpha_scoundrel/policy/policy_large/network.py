import torch
import torch.nn as nn
import torch.nn.functional as F
from scoundrel.rl.alpha_scoundrel.policy.policy_large.constants import (
    ACTION_SPACE,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_CARDS,
    STACK_SEQ_LEN,
    SCALAR_ENCODER_OUT,
)


class PolicyLargeNet(nn.Module):
    """
    Policy network with position-aware encoding for known dungeon cards.
    
    Key insight: In Scoundrel, when you "avoid" a room, those 4 cards go to the 
    back of the dungeon deck. The number of times avoided tells you how many 
    cards at the back you've seen, and they will appear in exact order (LIFO).
    
    This architecture respects that structure:
    - Unknown cards (front of deck): Encoded as aggregate statistics
    - Known cards (back of deck): Position-aware encoding preserves timing info
    
    "Turns until card appears" is critical - a monster-14 in 2 turns is very 
    different from monster-14 in 10 turns.
    """
    
    def __init__(
        self, 
        scalar_input_dim: int,
        embed_dim: int = None,
        hidden_dim: int = None,
        action_space: int = None,
        num_cards: int = None,
        stack_seq_len: int = None
    ):
        super(PolicyLargeNet, self).__init__()
        
        # Use provided constants or fall back to defaults
        embed_dim = embed_dim if embed_dim is not None else EMBED_DIM
        hidden_dim = hidden_dim if hidden_dim is not None else HIDDEN_DIM
        action_space = action_space if action_space is not None else ACTION_SPACE
        num_cards = num_cards if num_cards is not None else NUM_CARDS
        stack_seq_len = stack_seq_len if stack_seq_len is not None else STACK_SEQ_LEN
        
        # Card embedding (0 = unknown/padding)
        self.card_embed = nn.Embedding(num_cards, embed_dim, padding_idx=0)
        
        # Positional embedding: encodes "how many draws until this card appears"
        # Position 0 = next card drawn, position 39 = last card in deck
        self.pos_embed = nn.Embedding(stack_seq_len, embed_dim)
        
        # Per-card encoder: transforms (card_embed + pos_embed) -> features
        self.card_encoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # Unknown cards encoder: aggregate stats [potion_sum, weapon_sum, monster_sum]
        self.unknown_encoder = nn.Linear(3, 32)
        
        # Total cards encoder: aggregate stats for entire dungeon deck [potion_sum, weapon_sum, monster_sum]
        self.total_encoder = nn.Linear(3, 32)
        
        # Attention mechanism to learn which positions/cards matter most
        self.position_attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        
        # Scalar features encoder (room + status)
        self.scalar_encoder = nn.Linear(scalar_input_dim, 64)
        
        # Final policy head (10 FC layers)
        # 64 (known cards) + 32 (unknown stats) + 32 (total stats) + 64 (scalar) = 192
        self.scalar_fc = nn.Linear(192, SCALAR_ENCODER_OUT)
        self.shared_layer = nn.Linear(SCALAR_ENCODER_OUT, hidden_dim)
        self.fc_layer = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_layer7 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_space)

    def forward(self, scalar_data, sequence_data, unknown_stats, total_stats):
        """
        Forward pass through the network.
        
        Args:
            scalar_data: [batch_size, scalar_input_dim] tensor of room/status features
            sequence_data: [batch_size, seq_len] tensor of card IDs
                         (0 = unknown/padding, non-zero = known card)
            unknown_stats: [batch_size, 3] tensor of aggregate stats for unknown cards
                         [potion_sum, weapon_sum, monster_sum] (normalized)
            total_stats: [batch_size, 3] tensor of aggregate stats for entire dungeon deck
                         [potion_sum, weapon_sum, monster_sum] (normalized)
            
        Returns:
            logits: [batch_size, ACTION_SPACE] tensor of action logits
        """
        batch_size = sequence_data.size(0)
        seq_len = sequence_data.size(1)
        device = sequence_data.device
        
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed cards and positions
        card_emb = self.card_embed(sequence_data)  # [B, seq_len, embed_dim]
        pos_emb = self.pos_embed(positions)        # [B, seq_len, embed_dim]
        
        # Combine: card identity + positional info
        combined_emb = card_emb + pos_emb  # [B, seq_len, embed_dim]
        
        # Encode each position
        encoded = self.card_encoder(combined_emb)  # [B, seq_len, 64]
        
        # Mask for known cards only (non-zero IDs)
        known_mask = (sequence_data != 0).float().unsqueeze(-1)  # [B, seq_len, 1]
        
        # Attention-weighted aggregation of known cards
        attn_scores = self.position_attention(encoded)  # [B, seq_len, 1]
        attn_scores = attn_scores.masked_fill(known_mask == 0, float('-inf'))
        
        # Check if any batch has known cards
        has_known = (known_mask.sum(dim=1) > 0).float()  # [B, 1]
        
        # Softmax over sequence dimension (handle all-unknown case)
        # Replace -inf with 0 for batches with no known cards before softmax
        attn_scores_safe = torch.where(
            has_known.unsqueeze(1).expand_as(attn_scores) > 0,
            attn_scores,
            torch.zeros_like(attn_scores)
        )
        attn_weights = F.softmax(attn_scores_safe, dim=1)  # [B, seq_len, 1]
        
        # Zero out attention for batches with no known cards
        attn_weights = attn_weights * has_known.unsqueeze(1)
        
        known_features = (encoded * attn_weights).sum(dim=1)  # [B, 64]
        
        # Encode unknown card statistics
        unknown_features = F.relu(self.unknown_encoder(unknown_stats))  # [B, 32]
        
        # Encode total card statistics for entire dungeon deck
        total_features = F.relu(self.total_encoder(total_stats))  # [B, 32]
        
        # Encode scalar features (room + status)
        scalar_features = F.relu(self.scalar_encoder(scalar_data))  # [B, 64]
        
        # Combine all features
        combined = torch.cat([known_features, unknown_features, total_features, scalar_features], dim=1)
        
        # Pass through 10 FC layers
        x = F.relu(self.scalar_fc(combined))
        x = F.relu(self.shared_layer(x))
        x = F.relu(self.fc_layer(x))
        x = F.relu(self.fc_layer2(x))
        x = F.relu(self.fc_layer3(x))
        x = F.relu(self.fc_layer4(x))
        x = F.relu(self.fc_layer5(x))
        x = F.relu(self.fc_layer6(x))
        x = F.relu(self.fc_layer7(x))
        return self.action_head(x)
