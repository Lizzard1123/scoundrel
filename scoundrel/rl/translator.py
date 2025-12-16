import torch
from scoundrel.models.card import Card, CardType, Suit
from scoundrel.models.game_state import Action, GameState


class ScoundrelTranslator:
    """
    Bridges the Scoundrel Engine and RL agents.
    Converts game state to tensors and handles action encoding/decoding.
    """
    def __init__(self, stack_seq_len: int = 40):
        """
        Args:
            stack_seq_len: Maximum length for dungeon stack sequence (default 40)
                          Total deck is 44, but 4 cards dealt to room at start
        """
        self.stack_seq_len = stack_seq_len
        # Maps card types to IDs
        # 0: Pad
        # 1-13: Clubs (Mon): 2-14
        # 14-26: Spades (Mon): 2-14  
        # 27-35: Diamonds (Wep): 2-10
        # 36-44: Hearts (HP): 2-10
        self.suit_offsets = {
            Suit.CLUBS: 0,
            Suit.SPADES: 13,
            Suit.DIAMONDS: 26,
            Suit.HEARTS: 35
        }

    def _card_to_id(self, card: Card):
        if card is None:
            return 0
        val = card.value - 1  # Normalize 2-14 -> 1-13
        return self.suit_offsets[card.suit] + val

    def encode_state(self, game_state: GameState):
        """
        Converts raw engine state into PyTorch tensors.
        Returns (scalar_features, sequence_features)
        """
        # 1. SCALAR FEATURES (The "Situation")
        # Normalize simple stats to 0-1 range roughly
        hp = game_state.health / 20.0
        wep_val = 0
        if game_state.equipped_weapon:
            wep_val = game_state.equipped_weapon.value / 14.0
        wep_last = 0
        if len(game_state.weapon_monsters) > 0:
            wep_last = game_state.weapon_monsters[-1].value / 14.0
        can_run = 1.0 if game_state.can_avoid else 0.0
        can_heal = 1.0 if game_state.can_use_potion else 0.0

        # Encode Room Cards (Simplified: just values/types)
        room_vec = []
        for i in range(4):
            card = None
            if i >= len(game_state.room):
                room_vec.extend([0, 0, 0])  # Empty
                continue
            card = game_state.room[i]
            # Type: 0=Mon, 1=Wep, 2=HP
            c_type = 0 if card.type == CardType.MONSTER else (
                1 if card.type == CardType.WEAPON else 2
            )
            room_vec.extend([1, card.value/14.0, c_type])

        scalar_features = [hp, wep_val, wep_last, can_run, can_heal] + room_vec

        # 2. SEQUENCE FEATURES (The "Known Stack")
        stack_ids = []
        for i, elm in enumerate(game_state.dungeon):
            if i >= game_state.number_avoided * 4:
                stack_ids.append(self._card_to_id(elm))
            else:
                stack_ids.append(0)
        # Pad to fixed length
        padding = [0] * (self.stack_seq_len - len(stack_ids))
        stack_seq = stack_ids + padding

        return (
            torch.FloatTensor(scalar_features).unsqueeze(0),  # Batch dim
            torch.LongTensor(stack_seq).unsqueeze(0)
        )

    def get_action_mask(self, game_state: GameState):
        """
        Returns a boolean tensor mask [5] where False is invalid, True is valid.
        Order: [Pick1, Pick2, Pick3, Pick4, Run]
        """
        mask = [False] * 5

        # Check Pick actions
        for i in range(len(game_state.room)):
            mask[i] = True

        # Check Run action
        if game_state.can_avoid:
            mask[4] = True

        return torch.tensor(mask, dtype=torch.bool)

    def decode_action(self, action_idx: int):
        """Maps action index back to engine readable action."""
        if action_idx == 4:
            return Action.AVOID
        return list(Action)[action_idx]

