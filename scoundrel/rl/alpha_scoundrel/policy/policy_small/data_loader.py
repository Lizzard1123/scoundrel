import torch
from pathlib import Path
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from scoundrel.models.card import CardType
from scoundrel.models.game_state import GameState
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.utils import get_pin_memory
from scoundrel.rl.alpha_scoundrel.data_utils import (
    deserialize_game_state,
    visits_to_distribution,
)


def compute_stack_sums(game_state: GameState) -> torch.Tensor:
    """
    Compute sums of card values in dungeon stack by type.
    
    Args:
        game_state: Current game state
        
    Returns:
        Tensor [3] with [potion_sum, weapon_sum, monster_sum]
        Values are normalized by dividing by 14.0 (max card value)
    """
    potion_sum = 0.0
    weapon_sum = 0.0
    monster_sum = 0.0
    
    # Only count cards that haven't been avoided
    start_idx = game_state.number_avoided * 4
    for card in game_state.dungeon[start_idx:]:
        if card.type == CardType.POTION:
            potion_sum += card.value
        elif card.type == CardType.WEAPON:
            weapon_sum += card.value
        elif card.type == CardType.MONSTER:
            monster_sum += card.value
    
    # Normalize by max card value (14)
    return torch.FloatTensor([
        potion_sum / 14.0,
        weapon_sum / 14.0,
        monster_sum / 14.0
    ])


class MCTSDataset(Dataset):
    """
    Dataset for loading MCTS log files and converting to training samples.
    Uses regular game state features + dungeon stack sums instead of sequence.
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
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        
        log_files = sorted(self.log_dir.glob("*.json"))
        if max_games is not None:
            log_files = log_files[:max_games]
        
        print(f"Loading {len(log_files)} game log files...")
        
        import json
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    game_data = json.load(f)
                
                events = game_data.get("events", [])
                for event in events:
                    try:
                        game_state_dict = event.get("game_state", {})
                        game_state = deserialize_game_state(game_state_dict)
                        scalar_features, _ = translator.encode_state(game_state)
                        stack_sums = compute_stack_sums(game_state)
                        action_mask = translator.get_action_mask(game_state)
                        mcts_stats = event.get("mcts_stats", [])
                        target_probs = visits_to_distribution(mcts_stats, action_mask)
                        
                        self.samples.append((
                            scalar_features.squeeze(0),
                            stack_sums,
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
        scalar_features, stack_sums, target_probs, action_mask = self.samples[idx]
        return scalar_features, stack_sums, target_probs, action_mask


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

