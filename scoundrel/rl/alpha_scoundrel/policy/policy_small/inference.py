"""
Inference implementation for Policy Small model.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional, Dict

from scoundrel.models.game_state import Action, GameState
from scoundrel.rl.alpha_scoundrel.inference import BaseInference
from scoundrel.rl.alpha_scoundrel.policy.policy_small.network import PolicySmallNet
from scoundrel.rl.alpha_scoundrel.policy.policy_small.constants import (
    STACK_SEQ_LEN,
    SCALAR_ENCODER_OUT,
    HIDDEN_DIM,
    ACTION_SPACE
)
from scoundrel.rl.alpha_scoundrel.policy.policy_small.data_loader import compute_stack_sums
from scoundrel.rl.utils import mask_logits


class PolicySmallInference(BaseInference):
    """
    Inference class for Policy Small model.
    
    Automatically infers architecture constants from checkpoint state dict if not provided,
    making it robust to architecture changes over time.
    """
    
    def __init__(
        self,
        checkpoint_path: Path | str,
        scalar_input_dim: Optional[int] = None,
        device: Optional[str] = None,
        scalar_encoder_out: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        action_space: Optional[int] = None
    ):
        """
        Initialize Policy Small inference.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            scalar_input_dim: Scalar input dimension (auto-detected if None)
            device: Device to use ("cpu" or "cuda", auto-detected if None)
            scalar_encoder_out: Architecture constant SCALAR_ENCODER_OUT 
                               (auto-detected from checkpoint if None)
            hidden_dim: Architecture constant HIDDEN_DIM 
                       (auto-detected from checkpoint if None)
            action_space: Architecture constant ACTION_SPACE 
                         (auto-detected from checkpoint if None)
        """
        # Store checkpoint path for architecture inference
        self._checkpoint_path = Path(checkpoint_path)
        
        # If any architecture param is None, infer from checkpoint
        if scalar_encoder_out is None or hidden_dim is None or action_space is None:
            inferred = self._infer_architecture_from_checkpoint()
            scalar_encoder_out = scalar_encoder_out or inferred['scalar_encoder_out']
            hidden_dim = hidden_dim or inferred['hidden_dim']
            action_space = action_space or inferred['action_space']
        
        self.scalar_encoder_out = scalar_encoder_out
        self.hidden_dim = hidden_dim
        self.action_space = action_space
        
        super().__init__(
            checkpoint_path=checkpoint_path,
            stack_seq_len=STACK_SEQ_LEN,
            scalar_input_dim=scalar_input_dim,
            device=device
        )
    
    def _infer_architecture_from_checkpoint(self) -> Dict[str, int]:
        """
        Infer architecture constants from checkpoint state dict shapes.
        
        Returns:
            Dictionary with architecture constants:
            - scalar_encoder_out: Output dimension of scalar encoder
            - hidden_dim: Hidden layer dimension
            - action_space: Action space size
        """
        checkpoint = torch.load(self._checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # scalar_fc.weight: [scalar_encoder_out, combined_input_dim]
        scalar_encoder_out = state_dict['scalar_fc.weight'].shape[0]
        
        # shared_layer.weight: [hidden_dim, scalar_encoder_out]
        hidden_dim = state_dict['shared_layer.weight'].shape[0]
        
        # action_head.weight: [action_space, hidden_dim]
        action_space = state_dict['action_head.weight'].shape[0]
        
        return {
            'scalar_encoder_out': scalar_encoder_out,
            'hidden_dim': hidden_dim,
            'action_space': action_space,
        }
    
    def _load_model(self) -> PolicySmallNet:
        """Load PolicySmallNet model from checkpoint."""
        checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device)
        model = PolicySmallNet(
            scalar_input_dim=self.scalar_input_dim,
            scalar_encoder_out=self.scalar_encoder_out,
            hidden_dim=self.hidden_dim,
            action_space=self.action_space
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
        s_scal, _ = self.translator.encode_state(state)
        stack_sums = compute_stack_sums(state)
        mask = self.translator.get_action_mask(state)
        
        # Move to device
        s_scal = s_scal.to(self.device)
        stack_sums = stack_sums.to(self.device)
        mask = mask.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(s_scal, stack_sums.unsqueeze(0))
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

