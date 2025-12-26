"""
Base inference class for Alpha Scoundrel models.

This module provides a base class for loading and running inference with trained models.
Subclasses should inherit from this base class and implement the abstract methods.
"""

import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from scoundrel.models.game_state import GameState
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.game.game_manager import GameManager


class BaseInference(ABC):
    """
    Base class for model inference.
    
    Provides common functionality for loading models and running inference.
    Subclasses should implement model-specific forward pass logic.
    """
    
    def __init__(
        self,
        checkpoint_path: Path | str,
        stack_seq_len: int,
        scalar_input_dim: Optional[int] = None,
        device: Optional[str] = None
    ):
        """
        Initialize inference object.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            stack_seq_len: Sequence length for dungeon stack (typically 40)
            scalar_input_dim: Scalar input dimension. If None, will be auto-detected
            device: Device to use ("cpu" or "cuda", auto-detected if None)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.stack_seq_len = stack_seq_len
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Initialize translator
        self.translator = ScoundrelTranslator(stack_seq_len=stack_seq_len)
        
        # Try to get scalar_input_dim from checkpoint first
        if scalar_input_dim is None:
            checkpoint_data = torch.load(self.checkpoint_path, map_location=self.device)
            if isinstance(checkpoint_data, dict) and 'scalar_input_dim' in checkpoint_data:
                scalar_input_dim = checkpoint_data['scalar_input_dim']
        
        # Determine scalar_input_dim if still not provided
        if scalar_input_dim is None:
            scalar_input_dim = self._determine_scalar_input_dim()
        
        self.scalar_input_dim = scalar_input_dim
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        self.model = self.model.to(self.device)
    
    def _determine_scalar_input_dim(self) -> int:
        """
        Determine scalar input dimension by encoding a sample state.
        
        Returns:
            Scalar input dimension
        """
        temp_manager = GameManager()
        temp_state = temp_manager.restart()
        s_scal, _ = self.translator.encode_state(temp_state)
        return s_scal.shape[1]
    
    
    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """
        Load the model from checkpoint.
        
        Subclasses should implement this to create and load their specific model type.
        
        Returns:
            Loaded model (not yet moved to device or set to eval mode)
        """
        pass
    
    @abstractmethod
    def __call__(self, state: GameState):
        """
        Run inference on a game state.
        
        Subclasses should implement this to handle model-specific inference.
        
        Args:
            state: GameState object
            
        Returns:
            Model-specific output (type depends on subclass implementation)
        """
        pass
    
    def _load_state_dict(self, model: torch.nn.Module, checkpoint_data: dict):
        """
        Load state dict into model, handling different checkpoint formats.
        
        Args:
            model: Model to load state dict into
            checkpoint_data: Checkpoint data (dict or state dict)
        """
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        else:
            model.load_state_dict(checkpoint_data)

