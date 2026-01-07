"""
Reinforcement Learning Data Loader for Self-Play Training.

Loads game data for REINFORCE-style policy gradient training:
- Each sample is (state, action_taken, reward)
- Reward: +1 for wins, -1 for losses (or normalized score)
- Actions from winning games are reinforced, losing games discouraged
"""

import json
import torch
from pathlib import Path
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from scoundrel.models.card import CardType
from scoundrel.game.combat import Combat
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.utils import get_pin_memory
from scoundrel.rl.alpha_scoundrel.data_utils import deserialize_game_state


def compute_unknown_stats(game_state) -> torch.Tensor:
    """Compute aggregate statistics for unknown cards in dungeon."""
    known_count = game_state.number_avoided * 4
    if known_count > 0 and known_count < len(game_state.dungeon):
        unknown_cards = game_state.dungeon[:-known_count]
    elif known_count >= len(game_state.dungeon):
        unknown_cards = []
    else:
        unknown_cards = game_state.dungeon

    potion_sum = sum(c.value for c in unknown_cards if c.type == CardType.POTION)
    weapon_sum = sum(c.value for c in unknown_cards if c.type == CardType.WEAPON)
    monster_sum = sum(c.value for c in unknown_cards if c.type == CardType.MONSTER)

    return torch.tensor([
        potion_sum / 100.0,
        weapon_sum / 150.0,
        monster_sum / 250.0
    ], dtype=torch.float32)


def compute_total_stats(game_state) -> torch.Tensor:
    """Compute aggregate statistics for ALL cards in dungeon deck."""
    all_cards = game_state.dungeon

    potion_sum = sum(c.value for c in all_cards if c.type == CardType.POTION)
    weapon_sum = sum(c.value for c in all_cards if c.type == CardType.WEAPON)
    monster_sum = sum(c.value for c in all_cards if c.type == CardType.MONSTER)

    return torch.tensor([
        potion_sum / 100.0,
        weapon_sum / 150.0,
        monster_sum / 250.0
    ], dtype=torch.float32)


def compute_room_features(game_state) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rich features for each room card."""
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
            room_features.append([0.0] * 8)
            room_mask.append(True)

    return (
        torch.tensor(room_features, dtype=torch.float32),
        torch.tensor(room_mask, dtype=torch.bool)
    )


def compute_dungeon_len(game_state) -> torch.Tensor:
    """Compute actual dungeon length."""
    return torch.tensor(len(game_state.dungeon), dtype=torch.long)


def compute_reward(final_score: float, reward_type: str = "binary") -> float:
    """
    Compute reward from final game score.

    Args:
        final_score: Final game score
        reward_type: Type of reward computation
            - "binary": +1 for win, -1 for loss
            - "normalized": Score normalized to [-1, 1] range
            - "scaled": Score divided by max possible score

    Returns:
        Reward value
    """
    if reward_type == "binary":
        return 1.0 if final_score > 0 else -1.0
    elif reward_type == "normalized":
        # Normalize score to [-1, 1] range based on game bounds
        # Max Score: 30 (20 HP + 10 Potion Bonus)
        # Min "Alive" Score: -188 (20 HP - 208 Total Monster Value)
        # Scores below -188 (e.g. -300 for exit) are clamped to -1.0
        
        min_val = -188.0
        max_val = 30.0
        
        # Linear mapping: (score - min) / (max - min) * 2 - 1
        normalized = (final_score - min_val) / (max_val - min_val) * 2.0 - 1.0
        
        return max(-1.0, min(1.0, normalized))
    elif reward_type == "scaled":
        # Just scale by max possible score
        return final_score / 20.0
    else:
        return 1.0 if final_score > 0 else -1.0


class RLPolicyDataset(Dataset):
    """
    Dataset for REINFORCE-style policy gradient training.

    Each sample contains:
    - State features (same as supervised training)
    - Action taken (integer 0-4)
    - Reward (+1 for win, -1 for loss)
    - Action mask (valid actions)
    """

    def __init__(
        self,
        log_dir: Path,
        translator: ScoundrelTranslator,
        max_games: Optional[int] = None,
        reward_type: str = "binary",
    ):
        """
        Args:
            log_dir: Directory containing game log JSON files
            translator: ScoundrelTranslator for encoding states
            max_games: Maximum number of games to load (None = all)
            reward_type: How to compute rewards ("binary", "normalized", "scaled")
        """
        self.log_dir = Path(log_dir)
        self.translator = translator
        self.reward_type = reward_type
        self.samples: List[Tuple] = []

        log_files = sorted(self.log_dir.glob("*.json"))
        if max_games is not None:
            log_files = log_files[:max_games]

        print(f"Loading {len(log_files)} game log files for RL training...")
        print(f"  Reward type: {reward_type}")

        wins = 0
        losses = 0

        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    game_data = json.load(f)

                # Get final score and compute reward
                metadata = game_data.get("metadata", {})
                final_score = metadata.get("final_score", 0.0)
                reward = compute_reward(final_score, reward_type)

                if final_score > 0:
                    wins += 1
                else:
                    losses += 1

                events = game_data.get("events", [])
                for event in events:
                    try:
                        game_state_dict = event.get("game_state", {})
                        game_state = deserialize_game_state(game_state_dict)

                        # Get state encoding
                        scalar_features, sequence_features = translator.encode_state(game_state)
                        action_mask = translator.get_action_mask(game_state)

                        # Get action taken
                        action_taken = event.get("selected_action", 0)

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
                            torch.tensor(action_taken, dtype=torch.long),
                            torch.tensor(reward, dtype=torch.float32),
                            action_mask,
                        ))
                    except Exception as e:
                        print(f"Warning: Skipping event in {log_file.name}: {e}")
                        continue

            except Exception as e:
                print(f"Warning: Failed to load {log_file.name}: {e}")
                continue

        print(f"Loaded {len(self.samples)} RL training samples from {len(log_files)} games")
        print(f"  Wins: {wins}, Losses: {losses}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_rl_dataloaders(
    log_dir: Path,
    translator: ScoundrelTranslator,
    batch_size: int = 64,
    train_val_split: float = 0.9,
    max_games: Optional[int] = None,
    num_workers: int = 0,
    reward_type: str = "binary",
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for RL policy training.

    Args:
        log_dir: Directory containing game log JSON files
        translator: ScoundrelTranslator for encoding states
        batch_size: Batch size for training
        train_val_split: Fraction of data for training
        max_games: Maximum number of games to load (None = all)
        num_workers: Number of worker processes for data loading
        reward_type: How to compute rewards ("binary", "normalized", "scaled")

    Returns:
        Tuple of (train_loader, val_loader)
    """
    full_dataset = RLPolicyDataset(
        log_dir,
        translator,
        max_games=max_games,
        reward_type=reward_type,
    )

    if len(full_dataset) == 0:
        raise ValueError(
            f"No training samples found in {log_dir}. "
            f"Please ensure the directory contains game log JSON files."
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

