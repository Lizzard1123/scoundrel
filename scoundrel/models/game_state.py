from dataclasses import dataclass, field
from typing import List, Optional
from scoundrel.models.card import Card, CardType


@dataclass
class GameState:
    dungeon: List[Card] = field(default_factory=list)
    room: List[Card] = field(default_factory=list)
    discard: List[Card] = field(default_factory=list)
    equipped_weapon: Optional[Card] = None
    weapon_monsters: List[Card] = field(default_factory=list)
    used_potion: bool = False
    health: int = 20
    last_room_avoided: bool = False
    exit: bool = False
    
    @property
    def score(self) -> int:
        if self.exit:
            return -300
        score_val = 0
        if self.health <= 0:
            # Calculate negative score
            remaining_monsters = [
                c
                for c in self.dungeon + self.room
                if c.type == CardType.MONSTER
            ]
            score_val = self.health - sum(m.value for m in remaining_monsters)
        else:
            # Calculate positive score
            last_potion = next(
                (c for c in reversed(self.discard) if c.type == CardType.POTION), None
            )
            score_val = self.health
            if self.health == 20 and last_potion:
                score_val += last_potion.value
        return score_val

    @property
    def can_avoid(self) -> bool:
        return (not self.last_room_avoided) and len(self.room) == 4

    @property
    def can_use_potion(self) -> bool:
        return not self.used_potion

    @property
    def game_over(self) -> bool:
        return self.health <= 0 or (len(self.dungeon) == 0 and len(self.room) == 0)
