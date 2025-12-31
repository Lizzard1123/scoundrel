"""
Transformer-based Policy Network for Scoundrel.

Key architectural improvements over MLP:
1. Self-attention over room cards to model card interactions
2. Positional transformer over dungeon sequence for future planning
3. Cross-attention between room and dungeon for strategic reasoning
4. Rich feature encoding per card with contextual information
5. Pre-LN transformer with SwiGLU activations for stable deep training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from scoundrel.rl.alpha_scoundrel.policy.policy_large.constants import (
    ACTION_SPACE,
    EMBED_DIM,
    HIDDEN_DIM,
    NUM_CARDS,
    STACK_SEQ_LEN,
    DROPOUT_RATE,
    NUM_HEADS,
    NUM_TRANSFORMER_LAYERS,
    FF_DIM_MULTIPLIER,
)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for better sequence modeling.
    Encodes position by rotating the query/key vectors.
    
    Note: This implementation is disabled for simplicity.
    Using learned positional embeddings instead.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 64):
        super().__init__()
        self.dim = dim
        # Use half dim for sin/cos pairs
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute sin/cos for max sequence length
        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)  # [seq_len, half_dim]
        # Create full rotation (each pair rotates)
        self.register_buffer('cos', freqs.cos())  # [seq_len, half_dim]
        self.register_buffer('sin', freqs.sin())  # [seq_len, half_dim]
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary embedding to input tensor.
        
        Args:
            x: [B, seq_len, dim] tensor
            offset: position offset
            
        Returns:
            Rotated tensor of same shape
        """
        seq_len = x.shape[1]
        half_dim = self.dim // 2
        
        # Get sin/cos for this sequence
        cos = self.cos[offset:offset + seq_len]  # [seq_len, half_dim]
        sin = self.sin[offset:offset + seq_len]  # [seq_len, half_dim]
        
        # Split x into pairs for rotation
        x1, x2 = x[..., :half_dim], x[..., half_dim:]  # [B, seq_len, half_dim] each
        
        # Apply rotation: (x1 * cos - x2 * sin, x1 * sin + x2 * cos)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        
        return torch.cat([out1, out2], dim=-1)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (better than ReLU/GELU for transformers).
    From: GLU Variants Improve Transformer (Shazeer 2020)
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional rotary positional embeddings.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.1,
        use_rope: bool = False,
        max_seq_len: int = 64
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [B, T_q, D]
            key: [B, T_k, D]
            value: [B, T_k, D]
            mask: [T_q, T_k] or [B, T_q, T_k] attention mask
            key_padding_mask: [B, T_k] True for padding positions
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if enabled
        if self.use_rope:
            q = q.transpose(1, 2).reshape(B * self.num_heads, T_q, self.head_dim)
            k = k.transpose(1, 2).reshape(B * self.num_heads, T_k, self.head_dim)
            q = self.rope(q)
            k = self.rope(k)
            q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T_q, T_k]
        
        # Apply masks
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask, float('-inf'))
        
        if key_padding_mask is not None:
            # key_padding_mask: [B, T_k] -> [B, 1, 1, T_k]
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(expanded_mask, float('-inf'))
            
            # Check for rows where ALL keys are masked (would cause NaN in softmax)
            # all_masked: [B, 1, 1, 1] - True if all T_k positions are masked
            all_masked = key_padding_mask.all(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(2)
            # Replace -inf with 0 for these rows so softmax gives uniform distribution
            attn = torch.where(all_masked.expand_as(attn), torch.zeros_like(attn), attn)
        
        attn = F.softmax(attn, dim=-1)
        
        # Handle NaN from softmax (shouldn't happen now, but safety check)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, H, T_q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block with SwiGLU FFN.
    Pre-LN is more stable for deep networks.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        is_cross_attention: bool = False,
        max_seq_len: int = 64
    ):
        super().__init__()
        
        self.is_cross_attention = is_cross_attention
        
        # Pre-LN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads, dropout, 
            use_rope=use_rope and not is_cross_attention,
            max_seq_len=max_seq_len
        )
        
        if is_cross_attention:
            self.norm_kv = nn.LayerNorm(embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ff_dim, embed_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, D]
            context: Context for cross-attention [B, T_ctx, D] (only for cross-attention blocks)
            mask: Attention mask
            key_padding_mask: Padding mask for self-attention
            context_padding_mask: Padding mask for cross-attention context
        """
        # Self-attention or cross-attention
        if self.is_cross_attention and context is not None:
            normed_x = self.norm1(x)
            normed_ctx = self.norm_kv(context)
            attn_out = self.attn(normed_x, normed_ctx, normed_ctx, key_padding_mask=context_padding_mask)
        else:
            normed_x = self.norm1(x)
            attn_out = self.attn(normed_x, normed_x, normed_x, mask=mask, key_padding_mask=key_padding_mask)
        
        x = x + self.dropout(attn_out)
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class RoomEncoder(nn.Module):
    """
    Encodes room cards with rich features and self-attention.
    Room cards form a set where order doesn't matter, but card interactions do.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Rich card feature projection
        # Input features per card: [is_present, value/14, type_monster, type_weapon, type_potion,
        #                           can_weapon_beat, damage_if_fight, is_beneficial]
        self.card_proj = nn.Linear(8, embed_dim)
        
        # Positional encoding (simple learnable, since room is a set)
        self.pos_embed = nn.Parameter(torch.randn(1, 4, embed_dim) * 0.02)
        
        # Self-attention layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, use_rope=False)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, room_features: torch.Tensor, room_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            room_features: [B, 4, 8] rich features for each room card
            room_mask: [B, 4] True for empty slots
            
        Returns:
            room_encoded: [B, 4, D] encoded room cards
            room_pooled: [B, D] pooled room representation
        """
        B = room_features.shape[0]
        device = room_features.device
        
        # Project to embedding dimension
        x = self.card_proj(room_features)  # [B, 4, D]
        x = x + self.pos_embed
        
        # Check if any batch has valid cards
        has_valid = (~room_mask).any(dim=1)  # [B]
        any_has_valid = has_valid.any()
        
        if any_has_valid:
            # Self-attention with masking
            for layer in self.layers:
                x = layer(x, key_padding_mask=room_mask)
            
            x = self.norm(x)
            
            # Pool over valid cards (mean pooling with mask)
            valid_mask = (~room_mask).float().unsqueeze(-1)  # [B, 4, 1]
            num_valid = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
            pooled = (x * valid_mask).sum(dim=1) / num_valid.squeeze(-1)  # [B, D]
        else:
            # All slots empty (shouldn't happen in normal gameplay)
            x = self.norm(x)
            pooled = torch.zeros(B, x.shape[-1], device=device)
        
        return x, pooled


class DungeonEncoder(nn.Module):
    """
    Encodes dungeon sequence respecting information boundaries.
    
    CRITICAL: The agent does NOT know the order of unknown cards!
    - Unknown cards (front of deck): Only aggregate stats + count are known
    - Known cards (back of deck, from avoided rooms): Full identity + position known
    
    This encoder:
    1. Masks unknown card positions from attention (they're like padding for attention)
    2. Only applies positional encoding to KNOWN cards
    3. Encodes unknown cards via aggregate stats + count only
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        num_cards: int,
        max_seq_len: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Card embedding (0 = unknown/padding, will be masked)
        self.card_embed = nn.Embedding(num_cards, embed_dim, padding_idx=0)
        
        # Positional embedding for KNOWN cards only
        # Position 0 = first known card to appear, 1 = second, etc.
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        
        # Unknown cards encoder: [potion_sum, weapon_sum, monster_sum, num_unknown/40]
        # num_unknown tells us how many draws until known cards appear
        self.unknown_encoder = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # Self-attention layers for known cards only (no RoPE needed, we use learned pos)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, use_rope=False, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Attention pooling for aggregation
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pool_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.pool_norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        sequence_data: torch.Tensor,
        unknown_stats: torch.Tensor,
        dungeon_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence_data: [B, seq_len] card IDs (0 = unknown/padding)
            unknown_stats: [B, 3] aggregate stats for unknown cards
            dungeon_len: [B] actual dungeon length for each sample
            
        Returns:
            dungeon_encoded: [B, seq_len, D] encoded dungeon sequence
            dungeon_pooled: [B, D] pooled representation (known cards + unknown summary)
        """
        B, seq_len = sequence_data.shape
        device = sequence_data.device
        
        # Identify known vs unknown cards
        # Known cards have non-zero IDs, unknown cards have ID 0
        is_known = (sequence_data != 0)  # [B, seq_len]
        
        # Count unknown cards per batch (cards with ID 0 that are within dungeon_len)
        position_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        is_in_dungeon = position_indices < dungeon_len.unsqueeze(1)  # [B, seq_len]
        is_unknown_in_dungeon = (~is_known) & is_in_dungeon  # [B, seq_len]
        num_unknown = is_unknown_in_dungeon.sum(dim=1, keepdim=True).float()  # [B, 1]
        
        # Embed known cards only
        card_emb = self.card_embed(sequence_data)  # [B, seq_len, D]
        
        # Create relative positions for known cards only
        # For each known card, what position is it among known cards?
        # This requires computing cumulative sum of known cards up to each position
        known_cumsum = is_known.long().cumsum(dim=1) - 1  # [B, seq_len], -1 for 0-indexed
        known_cumsum = known_cumsum.clamp(min=0)  # Clamp negatives
        
        # Apply positional embedding only to known cards
        pos_emb = self.pos_embed(known_cumsum)  # [B, seq_len, D]
        
        # Zero out embeddings for unknown/padding positions
        known_mask_expanded = is_known.unsqueeze(-1).float()  # [B, seq_len, 1]
        x = (card_emb + pos_emb) * known_mask_expanded  # [B, seq_len, D]
        
        # Create attention mask: mask out unknown cards AND padding
        # True = masked (don't attend)
        padding_mask = ~is_in_dungeon  # Beyond dungeon length
        unknown_mask = is_unknown_in_dungeon  # Unknown cards within dungeon
        attention_mask = padding_mask | unknown_mask  # [B, seq_len]
        
        # Check if there are any known cards in each batch
        has_known = is_known.any(dim=1)  # [B]
        any_has_known = has_known.any()  # scalar
        
        # Only run attention if at least one sample has known cards
        if any_has_known:
            # Self-attention over known cards only
            for layer in self.layers:
                x = layer(x, key_padding_mask=attention_mask)
            
            x = self.norm(x)
            
            # Pool known cards via attention
            pool_query = self.pool_query.expand(B, -1, -1)
            known_pooled = self.pool_attn(pool_query, x, x, key_padding_mask=attention_mask)
            known_pooled = self.pool_norm(known_pooled).squeeze(1)  # [B, D]
            
            # Zero out pooled result for batches with no known cards
            known_pooled = known_pooled * has_known.unsqueeze(-1).float()
        else:
            # No known cards in any sample - skip attention entirely
            x = self.norm(x)
            known_pooled = torch.zeros(B, x.shape[-1], device=device)
        
        # Encode unknown cards as aggregate features
        # [potion_sum, weapon_sum, monster_sum, num_unknown/40]
        unknown_features = torch.cat([
            unknown_stats,
            num_unknown / 40.0  # Normalize by max dungeon size
        ], dim=1)  # [B, 4]
        unknown_encoded = self.unknown_encoder(unknown_features)  # [B, D]
        
        # Combine known card representation with unknown summary
        dungeon_pooled = known_pooled + unknown_encoded  # [B, D]
        
        return x, dungeon_pooled


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between room and dungeon.
    Allows room cards to attend to upcoming dungeon cards for strategic reasoning.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cross_attn = TransformerBlock(
            embed_dim, num_heads, ff_dim, dropout, is_cross_attention=True
        )
    
    def forward(
        self,
        room_encoded: torch.Tensor,
        dungeon_encoded: torch.Tensor,
        room_mask: torch.Tensor,
        dungeon_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Room attends to dungeon.
        
        Args:
            room_encoded: [B, 4, D]
            dungeon_encoded: [B, seq_len, D]
            room_mask: [B, 4] True for empty slots
            dungeon_padding_mask: [B, seq_len] True for padding
            
        Returns:
            room_updated: [B, 4, D]
        """
        return self.cross_attn(
            room_encoded,
            context=dungeon_encoded,
            key_padding_mask=room_mask,
            context_padding_mask=dungeon_padding_mask
)


class PolicyLargeNet(nn.Module):
    """
    Transformer-based policy network for Scoundrel.
    
    Architecture:
    1. Rich feature encoding for room cards
    2. Embedding-based encoding for dungeon sequence
    3. Self-attention within room and dungeon
    4. Cross-attention: room attends to dungeon
    5. Global context from scalar features
    6. Action head with per-action logits
    
    Key improvements:
    - Pre-LN architecture for stable training
    - SwiGLU activations (better than ReLU/GELU)
    - Rotary positional embeddings for dungeon sequence
    - Multi-head attention with proper masking
    - Rich feature encoding per card
    """
    
    def __init__(
        self, 
        scalar_input_dim: int,
        embed_dim: int = None,
        hidden_dim: int = None,
        action_space: int = None,
        num_cards: int = None,
        stack_seq_len: int = None,
        dropout_rate: float = None,
        num_heads: int = None,
        num_transformer_layers: int = None,
        ff_dim_multiplier: float = None
    ):
        super(PolicyLargeNet, self).__init__()
        
        # Use provided constants or fall back to defaults
        embed_dim = embed_dim if embed_dim is not None else EMBED_DIM
        hidden_dim = hidden_dim if hidden_dim is not None else HIDDEN_DIM
        action_space = action_space if action_space is not None else ACTION_SPACE
        num_cards = num_cards if num_cards is not None else NUM_CARDS
        stack_seq_len = stack_seq_len if stack_seq_len is not None else STACK_SEQ_LEN
        dropout_rate = dropout_rate if dropout_rate is not None else DROPOUT_RATE
        num_heads = num_heads if num_heads is not None else NUM_HEADS
        num_transformer_layers = num_transformer_layers if num_transformer_layers is not None else NUM_TRANSFORMER_LAYERS
        ff_dim_multiplier = ff_dim_multiplier if ff_dim_multiplier is not None else FF_DIM_MULTIPLIER
        
        ff_dim = int(embed_dim * ff_dim_multiplier)
        
        # Store for inference
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.num_cards = num_cards
        self.stack_seq_len = stack_seq_len
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        
        # Room encoder
        self.room_encoder = RoomEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            ff_dim=ff_dim,
            dropout=dropout_rate
        )
        
        # Dungeon encoder
        self.dungeon_encoder = DungeonEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            ff_dim=ff_dim,
            num_cards=num_cards,
            max_seq_len=stack_seq_len,
            dropout=dropout_rate
        )
        
        # Cross-attention: room attends to dungeon
        self.cross_attn = CrossAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate
        )
        
        # Scalar features encoder (game state: hp, weapon, etc.)
        # scalar_input_dim includes: hp, weapon_val, weapon_last_monster, can_run, can_heal
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # Total dungeon stats encoder
        self.total_stats_encoder = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # Combine all features
        # room_pooled (D) + dungeon_pooled (D) + scalar (D) + total_stats (D) = 4D
        combined_dim = embed_dim * 4
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, action_space),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
    
    def forward(
        self,
        scalar_data: torch.Tensor,
        sequence_data: torch.Tensor,
        unknown_stats: torch.Tensor,
        total_stats: torch.Tensor,
        room_features: Optional[torch.Tensor] = None,
        room_mask: Optional[torch.Tensor] = None,
        dungeon_len: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            scalar_data: [B, scalar_input_dim] game state features
            sequence_data: [B, seq_len] card IDs (0 = unknown/padding)
            unknown_stats: [B, 3] aggregate stats for unknown cards
            total_stats: [B, 3] aggregate stats for all dungeon cards
            room_features: [B, 4, 8] rich features for room cards (optional, computed from scalar if None)
            room_mask: [B, 4] True for empty room slots (optional)
            dungeon_len: [B] actual dungeon length (optional)
            
        Returns:
            logits: [B, ACTION_SPACE] action logits
        """
        B = scalar_data.shape[0]
        device = scalar_data.device
        
        # Extract room features from scalar_data if not provided
        if room_features is None:
            room_features, room_mask = self._extract_room_features(scalar_data)
        
        # Compute dungeon length from sequence if not provided
        if dungeon_len is None:
            # Count non-padding positions (sequence_data != 0 OR position < some threshold)
            # Actually, padding is at the end, so find where all zeros start
            # For simplicity, use sequence_data and count total - this is approximate
            dungeon_len = (sequence_data != 0).long().sum(dim=1)
            # If all zeros (no known cards), use a reasonable default based on unknown_stats
            dungeon_len = torch.where(
                dungeon_len == 0,
                torch.full_like(dungeon_len, 20),  # Reasonable default
                dungeon_len
            )
        
        # Encode room cards
        room_encoded, room_pooled = self.room_encoder(room_features, room_mask)
        
        # Encode dungeon sequence
        # Create full attention mask: mask unknown cards AND padding (not just padding)
        is_known = (sequence_data != 0)  # [B, seq_len]
        position_indices = torch.arange(sequence_data.shape[1], device=device).unsqueeze(0)
        is_in_dungeon = position_indices < dungeon_len.unsqueeze(1)
        # Full mask: True for positions that should be masked (unknown OR padding)
        dungeon_full_mask = ~(is_known & is_in_dungeon)  # Only attend to known cards in dungeon
        
        dungeon_encoded, dungeon_pooled = self.dungeon_encoder(sequence_data, unknown_stats, dungeon_len)
        
        # Cross-attention: room attends to KNOWN dungeon cards only
        # Check if there are any known cards to attend to
        has_any_known = is_known.any()
        if has_any_known:
            room_updated = self.cross_attn(room_encoded, dungeon_encoded, room_mask, dungeon_full_mask)
        else:
            # No known cards - skip cross-attention, just use room_encoded
            room_updated = room_encoded
        
        # Pool cross-attended room
        valid_mask = (~room_mask).float().unsqueeze(-1)
        num_valid = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        room_cross_pooled = (room_updated * valid_mask).sum(dim=1) / num_valid.squeeze(-1)
        
        # Combine room pooled with cross-attended version
        room_final = room_pooled + room_cross_pooled
        
        # Encode scalar features
        scalar_encoded = self.scalar_encoder(scalar_data)
        
        # Encode total stats
        total_encoded = self.total_stats_encoder(total_stats)
        
        # Combine all features
        combined = torch.cat([room_final, dungeon_pooled, scalar_encoded, total_encoded], dim=1)
        
        # Safety: replace any NaN with zeros to prevent gradient explosion
        combined = torch.where(torch.isnan(combined), torch.zeros_like(combined), combined)
        
        # Policy head
        logits = self.policy_head(combined)
        
        # Safety: replace any NaN in output
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        
        return logits
    
    def _extract_room_features(self, scalar_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract rich room features from scalar_data.
        
        scalar_data format: [hp, wep_val, wep_last, can_run, can_heal, 
                            c1_present, c1_val, c1_type, 
                            c2_present, c2_val, c2_type,
                            c3_present, c3_val, c3_type,
                            c4_present, c4_val, c4_type]
        
        Returns:
            room_features: [B, 4, 8] per-card features
            room_mask: [B, 4] True for empty slots
        """
        B = scalar_data.shape[0]
        device = scalar_data.device
        
        # Extract game state
        hp = scalar_data[:, 0:1]  # [B, 1]
        wep_val = scalar_data[:, 1:2]  # [B, 1]
        
        room_features_list = []
        room_mask_list = []
        
        for i in range(4):
            base_idx = 5 + i * 3
            is_present = scalar_data[:, base_idx:base_idx+1]  # [B, 1]
            card_val = scalar_data[:, base_idx+1:base_idx+2]  # [B, 1]
            card_type = scalar_data[:, base_idx+2:base_idx+3]  # [B, 1] (0=monster, 1=weapon, 2=potion)
            
            # Compute derived features
            is_monster = (card_type < 0.5).float()  # type 0
            is_weapon = ((card_type >= 0.5) & (card_type < 1.5)).float()  # type 1
            is_potion = (card_type >= 1.5).float()  # type 2
            
            # Can weapon beat this monster?
            can_weapon_beat = ((wep_val >= card_val) & is_monster.bool()).float()
            
            # Damage if fight (approximate)
            damage_if_fight = torch.where(
                is_monster.bool(),
                torch.where(can_weapon_beat.bool(), torch.zeros_like(card_val), card_val),
                torch.zeros_like(card_val)
            )
            
            # Is beneficial (weapon or usable potion)
            is_beneficial = (is_weapon + is_potion).clamp(0, 1)
            
            # Stack features: [is_present, value, is_monster, is_weapon, is_potion, 
            #                  can_weapon_beat, damage_if_fight, is_beneficial]
            card_features = torch.cat([
                is_present, card_val, is_monster, is_weapon, is_potion,
                can_weapon_beat, damage_if_fight * 14, is_beneficial  # Scale damage back
            ], dim=1)  # [B, 8]
            
            room_features_list.append(card_features)
            room_mask_list.append((is_present.squeeze(-1) < 0.5))  # True if empty
        
        room_features = torch.stack(room_features_list, dim=1)  # [B, 4, 8]
        room_mask = torch.stack(room_mask_list, dim=1)  # [B, 4]
        
        return room_features, room_mask
