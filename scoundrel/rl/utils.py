"""
Shared utility functions for RL agents and viewers.
Contains common functions used across different RL implementations.
"""
import torch
from pathlib import Path
from typing import Tuple
from scoundrel.models.game_state import Action


SCORE_MIN = -188  # Die at 0 HP with max monsters remaining
SCORE_MAX = 30    # 20 HP + max potion = 10
SCORE_RANGE = SCORE_MAX - SCORE_MIN  # 218


def normalize_score(score: int) -> float:
    """
    Normalize game score to [0, 1] range.
    
    Score bounds:
    - Min: -188 (die at 0 HP with max monsters remaining)
      It takes exactly 20 damage to die, leaving 208-20=188 monster value
      Score = 0 - 188 = -188
    - Max: 30 (20 HP + max potion = 10)
    - Range: 218
    
    Args:
        score: Raw game score
        
    Returns:
        Normalized score in [0, 1] range
    """
    return (score - SCORE_MIN) / SCORE_RANGE


def denormalize_score(normalized_score: float) -> int:
    """
    Convert normalized score back to raw game score.
    
    Args:
        normalized_score: Normalized score in [0, 1] range
        
    Returns:
        Raw game score
    """
    return int(normalized_score * SCORE_RANGE + SCORE_MIN)


def format_action(action: Action) -> str:
    """
    Convert Action enum to human-readable string.
    
    Args:
        action: Action enum value
        
    Returns:
        Human-readable action string (e.g., "avoid", "use 1", "use 2")
    """
    if action == Action.AVOID:
        return "avoid"
    if action in {Action.USE_1, Action.USE_2, Action.USE_3, Action.USE_4}:
        return f"use {action.value + 1}"
    return action.name.lower()


def get_device() -> torch.device:
    """
    Get the best available PyTorch device.
    Priority: MPS (Apple Silicon) > CUDA > CPU
    
    Returns:
        torch.device instance
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def get_pin_memory() -> bool:
    """
    Determine if pin_memory should be enabled for DataLoader.
    Only enable for CUDA (not MPS).
    
    Returns:
        True if pin_memory should be enabled
    """
    return torch.cuda.is_available()


def default_paths(base_dir: Path, checkpoint_name: str = "checkpoint.pt") -> Tuple[Path, Path]:
    """
    Get default paths for runs and checkpoints directories.
    
    Args:
        base_dir: Base directory (typically __file__.parent)
        checkpoint_name: Name of the checkpoint file
        
    Returns:
        Tuple of (runs_dir, checkpoint_path)
    """
    runs = base_dir / "runs"
    checkpoints = base_dir / "checkpoints" / checkpoint_name
    runs.mkdir(parents=True, exist_ok=True)
    checkpoints.parent.mkdir(parents=True, exist_ok=True)
    return runs, checkpoints


def mask_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """
    Apply action mask to logits (DRY: reusable across modules).
    
    Args:
        logits: Model output [batch_size, 5] or [1, 5]
        action_mask: Boolean mask [batch_size, 5] or [5] or [1, 5]
        
    Returns:
        Masked logits with invalid actions set to -inf
    """
    masked_logits = logits.clone()
    # Ensure action_mask has same number of dimensions as logits
    if action_mask.dim() < logits.dim():
        action_mask = action_mask.unsqueeze(0)
    masked_logits[~action_mask] = float('-inf')
    return masked_logits
