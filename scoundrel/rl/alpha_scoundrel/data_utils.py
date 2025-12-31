"""
Shared data loading utilities for alpha_scoundrel policies.
Contains functions for deserializing game states and creating datasets.
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from scoundrel.models.card import Card, Suit
from scoundrel.models.game_state import GameState
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.utils import get_pin_memory


def deserialize_card(card_dict: Dict) -> Card:
    """
    Convert JSON card dict to Card object.
    
    Args:
        card_dict: {"value": int, "suit": str}
        
    Returns:
        Card object
    """
    value = card_dict["value"]
    suit_str = card_dict["suit"]
    
    suit_map = {
        "♠": Suit.SPADES,
        "♥": Suit.HEARTS,
        "♦": Suit.DIAMONDS,
        "♣": Suit.CLUBS,
    }
    
    suit = suit_map.get(suit_str)
    if suit is None:
        raise ValueError(f"Unknown suit: {suit_str}")
    
    return Card(value=value, suit=suit)


def deserialize_game_state(state_dict: Dict) -> GameState:
    """
    Convert JSON game_state dict to GameState object.
    
    Args:
        state_dict: Dictionary from JSON log file
        
    Returns:
        GameState object
    """
    dungeon = [deserialize_card(c) for c in state_dict.get("dungeon", [])]
    room = [deserialize_card(c) for c in state_dict.get("room", [])]
    discard = [deserialize_card(c) for c in state_dict.get("discard", [])]
    
    equipped_weapon = None
    if state_dict.get("equipped_weapon") is not None:
        equipped_weapon = deserialize_card(state_dict["equipped_weapon"])
    
    weapon_monsters = [deserialize_card(c) for c in state_dict.get("weapon_monsters", [])]
    
    return GameState(
        dungeon=dungeon,
        room=room,
        discard=discard,
        equipped_weapon=equipped_weapon,
        weapon_monsters=weapon_monsters,
        used_potion=state_dict.get("used_potion", False),
        health=state_dict.get("health", 20),
        number_avoided=state_dict.get("number_avoided", 0),
        last_room_avoided=state_dict.get("last_room_avoided", False),
        exit=state_dict.get("exit", False),
    )


def visits_to_distribution(
    mcts_stats: List[Dict],
    action_mask: torch.Tensor,
    temperature: float = 1.0,
    use_q_weights: bool = False
) -> torch.Tensor:
    """
    Convert MCTS visit counts to probability distribution with optional sharpening.
    
    Args:
        mcts_stats: List of dicts with 'action', 'visits', and optionally 'avg_value'
        action_mask: Boolean tensor [5] indicating valid actions
        temperature: Temperature for distribution sharpening (< 1.0 sharpens toward one-hot,
                     > 1.0 smooths toward uniform). Default 1.0 = no change.
        use_q_weights: If True, weight visits by their Q-values (avg_value from MCTS).
                       This emphasizes actions that not only were explored but also 
                       had high value estimates.
        
    Returns:
        Target probability distribution [5]
    """
    visits = torch.zeros(5)
    q_values = torch.zeros(5)
    
    for stat in mcts_stats:
        action = stat['action']
        if 0 <= action < 5:
            visits[action] = stat['visits']
            q_values[action] = stat.get('avg_value', 0.0)
    
    total_visits = visits.sum()
    
    if total_visits > 0:
        if use_q_weights:
            # Weight visits by Q-value
            # Shift Q-values to be positive (add offset based on min)
            valid_q = q_values[visits > 0]
            if len(valid_q) > 0:
                q_min = valid_q.min()
                q_shifted = q_values - q_min + 0.1  # Ensure all positive
            else:
                q_shifted = torch.ones(5)
            
            # Weighted visits = visits * (shifted Q-value)
            weighted = visits * q_shifted
            
            # Apply temperature sharpening via log-softmax
            log_weighted = torch.log(weighted + 1e-8)
            probs = F.softmax(log_weighted / temperature, dim=-1)
        else:
            # Standard visit-based distribution with temperature sharpening
            log_visits = torch.log(visits + 1e-8)
            probs = F.softmax(log_visits / temperature, dim=-1)
    else:
        # No visits - uniform over valid actions
        probs = torch.ones(5) / 5
    
    # Apply action mask
    probs = probs * action_mask.float()
    prob_sum = probs.sum()
    
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        # Fallback: uniform over valid actions
        valid_count = action_mask.sum().item()
        if valid_count > 0:
            probs = action_mask.float() / valid_count
        else:
            probs = torch.ones(5) / 5
    
    return probs

