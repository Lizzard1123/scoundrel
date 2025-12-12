import torch
from scoundrel.models.card import Card, CardType, Suit
from scoundrel.models.game_state import Action, GameState
from scoundrel.rl.transformer_mcts.constants import STACK_SEQ_LEN

class ScoundrelTranslator:
    """
    Bridges the Scoundrel Engine and the PyTorch Model.
    Separates the 'Unknown Deck' (probabilities) from the 'Known Stack' (deterministic).
    """
    def __init__(self):
        # Maps card types to IDs for the Transformer
        # 0: Pad, 1-13: Clubs(Mon), 14-26: Spades(Mon), 27-39: Dia(Wep), 40-52: Hearts(HP)
        self.suit_offsets = {
            Suit.CLUBS: 0,
            Suit.SPADES: 13,
            Suit.DIAMONDS: 26,
            Suit.HEARTS: 39
        }

    def _card_to_id(self, card: Card):
        if card is None: return 0
        val = card.value - 1 # Normalize 2-14 -> 1-13
        return self.suit_offsets[card.suit] + val

    def encode_state(self, game_state: GameState):
        """
        Converts raw engine state into PyTorch tensors.
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
                room_vec.extend([0, 0, 0]) # Empty
                continue
            card = game_state.room[i]
            # Type: 0=Mon, 1=Wep, 2=HP
            c_type = 0 if card.type == CardType.MONSTER else (1 if card.type == CardType.WEAPON else 2)
            room_vec.extend([1, card.value/14.0, c_type])

        scalar_features = [hp, wep_val, wep_last, can_run, can_heal] + room_vec

        # 2. SEQUENCE FEATURES (The "Known Stack")
        # This is for the Transformer to plan ahead
        stack_ids = []
        for i, elm in enumerate(game_state.dungeon):
            if(i >= game_state.number_avoided * 4):
                stack_ids.append(self._card_to_id(elm))
            else:
                stack_ids.append(0)
        # stack_ids = [self._card_to_id(c) for c in game_state.dungeon]
        # Pad to fixed length
        padding = [0] * (STACK_SEQ_LEN - len(stack_ids))
        stack_seq = stack_ids + padding

        return (
            torch.FloatTensor(scalar_features).unsqueeze(0), # Batch dim
            torch.LongTensor(stack_seq).unsqueeze(0)
        )

    def get_action_mask(self, game_state: GameState):
        """
        Returns a boolean tensor mask [1, 5] where 0 is invalid, 1 is valid.
        Order: [Pick1, Pick2, Pick3, Pick4, Run]
        """
        mask = [0] * 5

        # Check Pick actions
        for i in range(len(game_state.room)):
            mask[i] = 1

        # Check Run action
        if game_state.can_avoid:
            mask[4] = 1

        return torch.tensor(mask, dtype=torch.bool)

    def decode_action(self, action_idx: int):
        """Maps model output index back to engine readable action."""
        if action_idx == 4:
            return Action.AVOID
        return list(Action)[action_idx]
