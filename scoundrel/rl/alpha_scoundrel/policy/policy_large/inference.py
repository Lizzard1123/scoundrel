"""
Inference implementation for Policy Large Transformer model.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional, Dict

from scoundrel.models.game_state import Action, GameState
from scoundrel.rl.alpha_scoundrel.inference import BaseInference
from scoundrel.rl.alpha_scoundrel.policy.policy_large.network import PolicyLargeNet
from scoundrel.rl.alpha_scoundrel.policy.policy_large.constants import (
    STACK_SEQ_LEN,
    EMBED_DIM,
    HIDDEN_DIM,
    ACTION_SPACE,
    NUM_CARDS,
    NUM_HEADS,
    NUM_TRANSFORMER_LAYERS,
)
from scoundrel.rl.alpha_scoundrel.policy.policy_large.data_loader import (
    compute_unknown_stats,
    compute_total_stats,
    compute_room_features,
    compute_dungeon_len,
)
from scoundrel.rl.utils import mask_logits


class PolicyLargeInference(BaseInference):
    """
    Inference class for Policy Large Transformer model.
    
    Automatically infers architecture constants from checkpoint state dict if not provided,
    making it robust to architecture changes over time.
    """
    
    def __init__(
        self,
        checkpoint_path: Path | str,
        scalar_input_dim: Optional[int] = None,
        device: Optional[str] = None,
        embed_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        action_space: Optional[int] = None,
        num_cards: Optional[int] = None,
        stack_seq_len: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_transformer_layers: Optional[int] = None
    ):
        """
        Initialize Policy Large Transformer inference.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            scalar_input_dim: Scalar input dimension (auto-detected if None)
            device: Device to use ("cpu" or "cuda", auto-detected if None)
            embed_dim: Architecture constant EMBED_DIM 
                      (auto-detected from checkpoint if None)
            hidden_dim: Architecture constant HIDDEN_DIM 
                       (auto-detected from checkpoint if None)
            action_space: Architecture constant ACTION_SPACE 
                         (auto-detected from checkpoint if None)
            num_cards: Architecture constant NUM_CARDS 
                      (auto-detected from checkpoint if None)
            stack_seq_len: Architecture constant STACK_SEQ_LEN 
                          (auto-detected from checkpoint if None)
            num_heads: Number of attention heads
            num_transformer_layers: Number of transformer layers
        """
        # Store checkpoint path for architecture inference
        self._checkpoint_path = Path(checkpoint_path)
        
        # If any architecture param is None, infer from checkpoint
        if any(p is None for p in [embed_dim, hidden_dim, action_space, num_cards, 
                                    stack_seq_len, num_heads, num_transformer_layers]):
            inferred = self._infer_architecture_from_checkpoint()
            embed_dim = embed_dim or inferred['embed_dim']
            hidden_dim = hidden_dim or inferred['hidden_dim']
            action_space = action_space or inferred['action_space']
            num_cards = num_cards or inferred['num_cards']
            stack_seq_len = stack_seq_len or inferred['stack_seq_len']
            num_heads = num_heads or inferred['num_heads']
            num_transformer_layers = num_transformer_layers or inferred['num_transformer_layers']
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.num_cards = num_cards
        self.stack_seq_len = stack_seq_len
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        
        super().__init__(
            checkpoint_path=checkpoint_path,
            stack_seq_len=self.stack_seq_len,
            scalar_input_dim=scalar_input_dim,
            device=device
        )
    
    def _infer_architecture_from_checkpoint(self) -> Dict[str, int]:
        """
        Infer architecture constants from checkpoint state dict shapes.
        
        Returns:
            Dictionary with architecture constants
        """
        checkpoint = torch.load(self._checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Infer from various layer shapes
        # dungeon_encoder.card_embed.weight: [num_cards, embed_dim]
        if 'dungeon_encoder.card_embed.weight' in state_dict:
            num_cards, embed_dim = state_dict['dungeon_encoder.card_embed.weight'].shape
        else:
            # Fallback for older architecture
            num_cards, embed_dim = state_dict.get('card_embed.weight', torch.zeros(45, 64)).shape
        
        # dungeon_encoder.pos_embed.weight: [stack_seq_len, embed_dim]
        if 'dungeon_encoder.pos_embed.weight' in state_dict:
            stack_seq_len = state_dict['dungeon_encoder.pos_embed.weight'].shape[0]
        else:
            stack_seq_len = state_dict.get('pos_embed.weight', torch.zeros(40, 64)).shape[0]
        
        # policy_head last layer: [action_space, hidden_dim // 2]
        # Find the action head layer
        action_space = 5  # Default
        hidden_dim = 512  # Default
        for key in state_dict:
            if 'policy_head' in key and 'weight' in key:
                shape = state_dict[key].shape
                if shape[0] == 5:  # Action space output
                    action_space = shape[0]
                    # Previous layer gives hidden_dim // 2
                elif len(shape) == 2 and shape[1] == 5:
                    pass  # This is bias
            if 'policy_head.0.weight' in key:
                # First policy head layer: [hidden_dim, combined_dim]
                hidden_dim = state_dict[key].shape[0]
        
        # Count transformer layers
        num_transformer_layers = 0
        for key in state_dict:
            if 'room_encoder.layers.' in key:
                layer_num = int(key.split('room_encoder.layers.')[1].split('.')[0])
                num_transformer_layers = max(num_transformer_layers, layer_num + 1)
        if num_transformer_layers == 0:
            num_transformer_layers = 3  # Default
        
        # Infer num_heads from attention layer
        num_heads = 8  # Default
        for key in state_dict:
            if 'attn.q_proj.weight' in key:
                # q_proj.weight: [embed_dim, embed_dim]
                # num_heads = embed_dim // head_dim
                # We assume head_dim = 16 (common for small models)
                num_heads = max(1, embed_dim // 16)
                break
        
        return {
            'embed_dim': embed_dim,
            'hidden_dim': hidden_dim,
            'action_space': action_space,
            'num_cards': num_cards,
            'stack_seq_len': stack_seq_len,
            'num_heads': num_heads,
            'num_transformer_layers': num_transformer_layers,
        }
    
    def _load_model(self) -> PolicyLargeNet:
        """Load PolicyLargeNet model from checkpoint."""
        checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device)
        model = PolicyLargeNet(
            scalar_input_dim=self.scalar_input_dim,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            action_space=self.action_space,
            num_cards=self.num_cards,
            stack_seq_len=self.stack_seq_len,
            num_heads=self.num_heads,
            num_transformer_layers=self.num_transformer_layers
        )
        self._load_state_dict(model, checkpoint_data)
        return model
    
    def __call__(self, state: GameState) -> Tuple[Action, torch.Tensor]:
        """
        Run inference on a game state.
        
        Args:
            state: GameState object
            
        Returns:
            Tuple of (action_enum, action_probs)
            - action_enum: The selected Action enum
            - action_probs: Tensor of action probabilities [5]
        """
        # Encode state
        s_scal, s_seq = self.translator.encode_state(state)
        unknown_stats = compute_unknown_stats(state)
        total_stats = compute_total_stats(state)
        room_features, room_mask = compute_room_features(state)
        dungeon_len = compute_dungeon_len(state)
        mask = self.translator.get_action_mask(state)
        
        # Move to device
        s_scal = s_scal.to(self.device)
        s_seq = s_seq.to(self.device)
        unknown_stats = unknown_stats.unsqueeze(0).to(self.device)
        total_stats = total_stats.unsqueeze(0).to(self.device)
        room_features = room_features.unsqueeze(0).to(self.device)
        room_mask = room_mask.unsqueeze(0).to(self.device)
        dungeon_len = dungeon_len.unsqueeze(0).to(self.device)
        mask = mask.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(
                s_scal, s_seq, unknown_stats, total_stats,
                room_features=room_features, room_mask=room_mask, dungeon_len=dungeon_len
            )
            masked_logits = mask_logits(logits, mask)
            probs = F.softmax(masked_logits, dim=-1)
            action_idx = int(torch.argmax(probs).item())
        
        action_enum = self.translator.decode_action(action_idx)
        return action_enum, probs.squeeze(0).cpu()
    
    def get_action(self, state: GameState) -> Action:
        """Get just the action (without probabilities)."""
        action, _ = self(state)
        return action
    
    def get_probs(self, state: GameState) -> torch.Tensor:
        """Get just the action probabilities."""
        _, probs = self(state)
        return probs
