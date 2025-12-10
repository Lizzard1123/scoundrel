from enum import Enum
from dataclasses import dataclass


class Suit(Enum):
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


class CardType(Enum):
    MONSTER = "Monster"
    WEAPON = "Weapon"
    POTION = "Potion"

CardAction = {
    CardType.MONSTER: "Fight ",
    CardType.WEAPON:  "Take  ",
    CardType.POTION:  "Heal  ",
}

CardColor = {
    CardType.MONSTER: "red",
    CardType.WEAPON: "yellow",
    CardType.POTION: "green",
}
@dataclass
class Card:
    value: int
    suit: Suit

    @property
    def type(self) -> CardType:
        if self.suit in [Suit.CLUBS, Suit.SPADES]:
            return CardType.MONSTER
        elif self.suit == Suit.DIAMONDS:
            return CardType.WEAPON
        else:  # Hearts
            return CardType.POTION

    @property
    def display_value(self) -> str:
        if self.value == 14:
            return "A"
        elif self.value == 11:
            return "J"
        elif self.value == 12:
            return "Q"
        elif self.value == 13:
            return "K"
        return str(self.value)

    def __str__(self) -> str:
        return f"{self.display_value}{self.suit.value}"
