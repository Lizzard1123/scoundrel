from dataclasses import dataclass, field
from typing import List, Optional
from scoundrel.models.card import Card


@dataclass
class GameState:
    dungeon: List[Card] = field(default_factory=list)
    room: List[Card] = field(default_factory=list)
    discard: List[Card] = field(default_factory=list)
    equipped_weapon: Optional[Card] = None
    weapon_monsters: List[Card] = field(default_factory=list)
    health: int = 20
    last_room_avoided: bool = False

    @property
    def game_over(self) -> bool:
        return self.health <= 0 or (len(self.dungeon) == 0 and len(self.room) == 0)
