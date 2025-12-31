"""
Data loader for Policy Large Transformer Network.

Computes rich features for training including:
- Per-card room features with contextual information
- Dungeon sequence with known/unknown card encoding
- Game state statistics
"""

import json
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from scoundrel.models.card import Card, CardType
from scoundrel.game.combat import Combat
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.utils import get_pin_memory
from scoundrel.rl.alpha_scoundrel.data_utils import (
    deserialize_game_state,
    visits_to_distribution,
)


def compute_unknown_stats(game_state) -> torch.Tensor:
    """
    Compute aggregate statistics for unknown cards in dungeon.
    
    Unknown cards are the ones at the front of the dungeon that haven't
    been seen yet (before the avoided cards that went to the back).
    
    Args:
        game_state: GameState object
        
    Returns:
        Tensor [3] of [potion_sum, weapon_sum, monster_sum] normalized
    """
    # Known cards are at the BACK (avoided rooms), unknown are at the FRONT
    known_count = game_state.number_avoided * 4
    if known_count > 0 and known_count < len(game_state.dungeon):
        unknown_cards = game_state.dungeon[:-known_count]
    elif known_count >= len(game_state.dungeon):
        # All cards are known (all from avoided rooms)
        unknown_cards = []
    else:
        # No avoids, all cards are unknown
        unknown_cards = game_state.dungeon
    
    potion_sum = sum(c.value for c in unknown_cards if c.type == CardType.POTION)
    weapon_sum = sum(c.value for c in unknown_cards if c.type == CardType.WEAPON)
    monster_sum = sum(c.value for c in unknown_cards if c.type == CardType.MONSTER)
    
    # Normalize by reasonable max values
    # Max possible: all 9 potions (2-10) = 54, all 13 weapons (2-14) = 104, all 26 monsters = 208
    return torch.tensor([
        potion_sum / 100.0,
        weapon_sum / 150.0,
        monster_sum / 250.0
    ], dtype=torch.float32)


def compute_total_stats(game_state) -> torch.Tensor:
    """
    Compute aggregate statistics for ALL cards in dungeon deck.
    
    This includes both known and unknown cards - the entire dungeon deck.
    
    Args:
        game_state: GameState object
        
    Returns:
        Tensor [3] of [potion_sum, weapon_sum, monster_sum] normalized
    """
    # Use all cards in the dungeon deck
    all_cards = game_state.dungeon
    
    potion_sum = sum(c.value for c in all_cards if c.type == CardType.POTION)
    weapon_sum = sum(c.value for c in all_cards if c.type == CardType.WEAPON)
    monster_sum = sum(c.value for c in all_cards if c.type == CardType.MONSTER)
    
    # Normalize by reasonable max values
    return torch.tensor([
        potion_sum / 100.0,
        weapon_sum / 150.0,
        monster_sum / 250.0
    ], dtype=torch.float32)


def compute_room_features(game_state) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rich features for each room card.
    
    Features per card (8 total):
    - is_present: 1 if card exists, 0 otherwise
    - value: card value normalized by 14
    - is_monster: 1 if monster, 0 otherwise
    - is_weapon: 1 if weapon, 0 otherwise
    - is_potion: 1 if potion, 0 otherwise
    - can_weapon_beat: 1 if equipped weapon can beat this monster (value >= monster)
    - damage_if_fight: expected damage if fighting this card (normalized)
    - is_beneficial: 1 if weapon or potion (positive effect)
    
    Args:
        game_state: GameState object
        
    Returns:
        room_features: Tensor [4, 8] of per-card features
        room_mask: Tensor [4] boolean mask, True for empty slots
    """
    room_features = []
    room_mask = []
    
    weapon = game_state.equipped_weapon
    weapon_val = weapon.value if weapon else 0
    
    for i in range(4):
        if i < len(game_state.room):
            card = game_state.room[i]
            is_present = 1.0
            value = card.value / 14.0
            
            is_monster = 1.0 if card.type == CardType.MONSTER else 0.0
            is_weapon = 1.0 if card.type == CardType.WEAPON else 0.0
            is_potion = 1.0 if card.type == CardType.POTION else 0.0
            
            # Can weapon beat this monster?
            can_weapon_beat = 0.0
            damage_if_fight = 0.0
            if card.type == CardType.MONSTER:
                if Combat.can_use_weapon(game_state, card):
                    damage = Combat.calculate_damage(card, weapon)
                    can_weapon_beat = 1.0 if damage == 0 else 0.0
                    damage_if_fight = damage / 14.0
                else:
                    damage_if_fight = card.value / 14.0
            
            is_beneficial = 1.0 if card.type in [CardType.WEAPON, CardType.POTION] else 0.0
            
            features = [is_present, value, is_monster, is_weapon, is_potion,
                       can_weapon_beat, damage_if_fight, is_beneficial]
            room_features.append(features)
            room_mask.append(False)
        else:
            # Empty slot
            room_features.append([0.0] * 8)
            room_mask.append(True)
    
    return (
        torch.tensor(room_features, dtype=torch.float32),
        torch.tensor(room_mask, dtype=torch.bool)
    )


def compute_dungeon_len(game_state) -> torch.Tensor:
    """
    Compute actual dungeon length.
    
    Args:
        game_state: GameState object
        
    Returns:
        Tensor scalar with dungeon length
    """
    return torch.tensor(len(game_state.dungeon), dtype=torch.long)


class MCTSDataset(Dataset):
    """
    Dataset for loading MCTS log files and converting to training samples.
    
    Enhanced for transformer architecture with rich room features.
    Supports temperature sharpening and Q-value weighting for target distributions.
    """
    
    def __init__(
        self,
        log_dir: Path,
        translator: ScoundrelTranslator,
        max_games: Optional[int] = None,
        temperature: float = 1.0,
        use_q_weights: bool = False,
    ):
        """
        Args:
            log_dir: Directory containing MCTS log JSON files
            translator: ScoundrelTranslator for encoding states
            max_games: Maximum number of games to load (None = all)
            temperature: Temperature for sharpening target distributions.
                        < 1.0 sharpens toward one-hot (more decisive targets)
                        > 1.0 smooths toward uniform
                        1.0 = no change (default)
            use_q_weights: If True, weight visits by their Q-values from MCTS
        """
        self.log_dir = Path(log_dir)
        self.translator = translator
        self.temperature = temperature
        self.use_q_weights = use_q_weights
        self.samples: List[Tuple[
            torch.Tensor,  # scalar_features
            torch.Tensor,  # sequence_features
            torch.Tensor,  # unknown_stats
            torch.Tensor,  # total_stats
            torch.Tensor,  # room_features
            torch.Tensor,  # room_mask
            torch.Tensor,  # dungeon_len
            torch.Tensor,  # target_probs
            torch.Tensor,  # action_mask
        ]] = []
        
        log_files = sorted(self.log_dir.glob("*.json"))
        if max_games is not None:
            log_files = log_files[:max_games]
        
        print(f"Loading {len(log_files)} game log files...")
        print(f"  Temperature: {temperature}, Use Q-weights: {use_q_weights}")
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    game_data = json.load(f)
                
                events = game_data.get("events", [])
                for event in events:
                    try:
                        game_state_dict = event.get("game_state", {})
                        game_state = deserialize_game_state(game_state_dict)
                        scalar_features, sequence_features = translator.encode_state(game_state)
                        action_mask = translator.get_action_mask(game_state)
                        mcts_stats = event.get("mcts_stats", [])
                        
                        # Apply temperature sharpening and Q-value weighting
                        target_probs = visits_to_distribution(
                            mcts_stats, 
                            action_mask,
                            temperature=temperature,
                            use_q_weights=use_q_weights
                        )
                        
                        # Compute enhanced features
                        unknown_stats = compute_unknown_stats(game_state)
                        total_stats = compute_total_stats(game_state)
                        room_features, room_mask = compute_room_features(game_state)
                        dungeon_len = compute_dungeon_len(game_state)
                        
                        self.samples.append((
                            scalar_features.squeeze(0),
                            sequence_features.squeeze(0),
                            unknown_stats,
                            total_stats,
                            room_features,
                            room_mask,
                            dungeon_len,
                            target_probs,
                            action_mask
                        ))
                    except Exception as e:
                        print(f"Warning: Skipping event in {log_file.name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Warning: Failed to load {log_file.name}: {e}")
                continue
        
        print(f"Loaded {len(self.samples)} training samples from {len(log_files)} games")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloaders(
    log_dir: Path,
    translator: ScoundrelTranslator,
    batch_size: int = 64,
    train_val_split: float = 0.9,
    max_games: Optional[int] = None,
    num_workers: int = 0,
    temperature: float = 1.0,
    use_q_weights: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from MCTS logs.
    
    Args:
        log_dir: Directory containing MCTS log JSON files
        translator: ScoundrelTranslator for encoding states
        batch_size: Batch size for training
        train_val_split: Fraction of data for training (rest is validation)
        max_games: Maximum number of games to load (None = all)
        num_workers: Number of worker processes for data loading
        temperature: Temperature for sharpening target distributions (< 1.0 sharpens)
        use_q_weights: If True, weight visits by their Q-values from MCTS
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    full_dataset = MCTSDataset(
        log_dir, 
        translator, 
        max_games=max_games,
        temperature=temperature,
        use_q_weights=use_q_weights
    )
    
    if len(full_dataset) == 0:
        raise ValueError(
            f"No training samples found in {log_dir}. "
            f"Please ensure the directory contains MCTS log JSON files."
        )
    
    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=get_pin_memory(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=get_pin_memory(),
    )
    
    return train_loader, val_loader
