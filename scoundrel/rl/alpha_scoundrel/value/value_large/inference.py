"""
Inference implementation for Value Large model.
"""

import torch
from pathlib import Path
from typing import Optional

from scoundrel.models.game_state import GameState
from scoundrel.rl.alpha_scoundrel.inference import BaseInference
from scoundrel.rl.alpha_scoundrel.value.value_large.network import ValueLargeNet
from scoundrel.rl.alpha_scoundrel.value.value_large.constants import STACK_SEQ_LEN
from scoundrel.rl.alpha_scoundrel.value.value_large.data_loader import compute_unknown_stats


class ValueLargeInference(BaseInference):
    """
    Inference class for Value Large model.
    """
    
    def __init__(
        self,
        checkpoint_path: Path | str,
        scalar_input_dim: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Value Large inference.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            scalar_input_dim: Scalar input dimension (auto-detected if None)
            device: Device to use ("cpu" or "cuda", auto-detected if None)
        """
        super().__init__(
            checkpoint_path=checkpoint_path,
            stack_seq_len=STACK_SEQ_LEN,
            scalar_input_dim=scalar_input_dim,
            device=device
        )
    
    def _load_model(self) -> ValueLargeNet:
        """Load ValueLargeNet model from checkpoint."""
        checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device)
        model = ValueLargeNet(scalar_input_dim=self.scalar_input_dim)
        self._load_state_dict(model, checkpoint_data)
        return model
    
    def __call__(self, state: GameState) -> float:
        """
        Run inference on a game state.
        
        Args:
            state: GameState object
            
        Returns:
            Predicted final game score (float)
        """
        # Encode state
        s_scal, s_seq = self.translator.encode_state(state)
        unknown_stats = compute_unknown_stats(state)
        
        # Move to device
        s_scal = s_scal.to(self.device)
        s_seq = s_seq.to(self.device)
        unknown_stats = unknown_stats.to(self.device)
        
        # Run inference
        with torch.no_grad():
            value = self.model(s_scal, s_seq, unknown_stats.unsqueeze(0))
            predicted_value = value.item()
        
        return predicted_value
    
    def get_value(self, state: GameState) -> float:
        """Get predicted value (alias for __call__)."""
        return self(state)

