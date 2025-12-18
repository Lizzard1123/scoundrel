from typing import List, Optional
import random
from scoundrel.models.card import Card, Suit


class Deck:
    @staticmethod
    def create_deck(seed: Optional[int] = None) -> List[Card]:
        """
        Create and shuffle a deck of cards.
        
        Args:
            seed: Optional seed for deterministic deck shuffling.
                  If provided, the same seed will produce the same deck order.
                  If None, deck order is random.
        
        Returns:
            List of shuffled Card objects
        """
        cards = []
        for suit in Suit:
            # Skip red face cards and aces as per rules
            if suit in [Suit.HEARTS, Suit.DIAMONDS]:
                values = range(2, 11)
            else:
                values = range(2, 15)

            for value in values:
                cards.append(Card(value, suit))

        if seed is not None:
            random.seed(seed)
        random.shuffle(cards)
        return cards
