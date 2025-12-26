"""
Inference implementation for Policy Large model.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional

from scoundrel.models.game_state import Action, GameState
from scoundrel.rl.alpha_scoundrel.inference import BaseInference
from scoundrel.rl.alpha_scoundrel.policy.policy_large.network import PolicyLargeNet
from scoundrel.rl.alpha_scoundrel.policy.policy_large.constants import STACK_SEQ_LEN
from scoundrel.rl.alpha_scoundrel.policy.policy_large.data_loader import compute_unknown_stats
from scoundrel.rl.utils import mask_logits


class PolicyLargeInference(BaseInference):
    """
    Inference class for Policy Large model.
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
        stack_seq_len: Optional[int] = None
    ):
        """
        Initialize Policy Large inference.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            scalar_input_dim: Scalar input dimension (auto-detected if None)
            device: Device to use ("cpu" or "cuda", auto-detected if None)
            embed_dim: Architecture constant EMBED_DIM (uses default if None)
            hidden_dim: Architecture constant HIDDEN_DIM (uses default if None)
            action_space: Architecture constant ACTION_SPACE (uses default if None)
            num_cards: Architecture constant NUM_CARDS (uses default if None)
            stack_seq_len: Architecture constant STACK_SEQ_LEN (uses default if None)
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        self.num_cards = num_cards
        self.stack_seq_len = stack_seq_len if stack_seq_len is not None else STACK_SEQ_LEN
        
        super().__init__(
            checkpoint_path=checkpoint_path,
            stack_seq_len=self.stack_seq_len,
            scalar_input_dim=scalar_input_dim,
            device=device
        )
    
    def _load_model(self) -> PolicyLargeNet:
        """Load PolicyLargeNet model from checkpoint."""
        checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device)
        model = PolicyLargeNet(
            scalar_input_dim=self.scalar_input_dim,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            action_space=self.action_space,
            num_cards=self.num_cards,
            stack_seq_len=self.stack_seq_len
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
        mask = self.translator.get_action_mask(state)
        
        # Move to device
        s_scal = s_scal.to(self.device)
        s_seq = s_seq.to(self.device)
        unknown_stats = unknown_stats.to(self.device)
        mask = mask.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(s_scal, s_seq, unknown_stats.unsqueeze(0))
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

