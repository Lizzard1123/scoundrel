from typing import List
import random
from scoundrel.models.card import Card, Suit


class Deck:
    @staticmethod
    def create_deck() -> List[Card]:
        cards = []
        for suit in Suit:
            # Skip red face cards and aces as per rules
            if suit in [Suit.HEARTS, Suit.DIAMONDS]:
                values = range(2, 11)
            else:
                values = range(2, 15)

            for value in values:
                cards.append(Card(value, suit))

        random.shuffle(cards)
        return cards
