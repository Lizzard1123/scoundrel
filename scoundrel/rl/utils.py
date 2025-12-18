"""
Shared utility functions for RL agents and viewers.
Contains common functions used across different RL implementations.
"""
from scoundrel.models.game_state import Action


# Score normalization constants
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
    return int(normalized_score * SCORE_RANGE - SCORE_MIN)


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
