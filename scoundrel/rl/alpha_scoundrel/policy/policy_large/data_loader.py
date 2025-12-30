import json
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from scoundrel.models.card import CardType
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
    # Max possible: all 9 potions (2-10) = 54, all 13 weapons (2-14) = 104, all 26 monsters = 208
    return torch.tensor([
        potion_sum / 100.0,
        weapon_sum / 150.0,
        monster_sum / 250.0
    ], dtype=torch.float32)


class MCTSDataset(Dataset):
    """
    Dataset for loading MCTS log files and converting to training samples.
    """
    
    def __init__(
        self,
        log_dir: Path,
        translator: ScoundrelTranslator,
        max_games: Optional[int] = None,
    ):
        """
        Args:
            log_dir: Directory containing MCTS log JSON files
            translator: ScoundrelTranslator for encoding states
            max_games: Maximum number of games to load (None = all)
        """
        self.log_dir = Path(log_dir)
        self.translator = translator
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        
        log_files = sorted(self.log_dir.glob("*.json"))
        if max_games is not None:
            log_files = log_files[:max_games]
        
        print(f"Loading {len(log_files)} game log files...")
        
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
                        target_probs = visits_to_distribution(mcts_stats, action_mask)
                        
                        # Compute unknown card statistics
                        unknown_stats = compute_unknown_stats(game_state)
                        # Compute total card statistics for entire dungeon deck
                        total_stats = compute_total_stats(game_state)
                        
                        self.samples.append((
                            scalar_features.squeeze(0),
                            sequence_features.squeeze(0),
                            unknown_stats,
                            total_stats,
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
        scalar_features, sequence_features, unknown_stats, total_stats, target_probs, action_mask = self.samples[idx]
        return scalar_features, sequence_features, unknown_stats, total_stats, target_probs, action_mask


def create_dataloaders(
    log_dir: Path,
    translator: ScoundrelTranslator,
    batch_size: int = 64,
    train_val_split: float = 0.9,
    max_games: Optional[int] = None,
    num_workers: int = 0,
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
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    full_dataset = MCTSDataset(log_dir, translator, max_games=max_games)
    
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

