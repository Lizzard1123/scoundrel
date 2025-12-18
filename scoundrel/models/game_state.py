from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from scoundrel.models.card import Card, CardType

class Action(Enum):
    USE_1 = 0
    USE_2 = 1
    USE_3 = 2
    USE_4 = 3
    RESTART = 4
    EXIT = 5
    INVALID = 6
    AVOID = 7
@dataclass
class GameState:
    dungeon: List[Card] = field(default_factory=list)
    room: List[Card] = field(default_factory=list)
    discard: List[Card] = field(default_factory=list)
    equipped_weapon: Optional[Card] = None
    weapon_monsters: List[Card] = field(default_factory=list)
    used_potion: bool = False
    health: int = 20
    number_avoided: int = 0
    last_room_avoided: bool = False
    exit: bool = False

    @property
    def score(self) -> int:
        if self.exit:
            return -300
        score_val = 0
        if self.health <= 0:
            remaining_monsters = [
                c
                for c in self.dungeon + self.room
                if c.type == CardType.MONSTER
            ]
            score_val = self.health - sum(m.value for m in remaining_monsters)
        else:
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
    def lost(self) -> bool:
        return self.health <= 0 or (len(self.dungeon) == 0 and len(self.room) == 0)

    @property
    def game_over(self) -> bool:
        return self.lost or self.exit
    
    def copy(self) -> 'GameState':
        """
        Create an optimized copy of the GameState.
        
        Uses shallow copy for the GameState object and copies only mutable lists.
        Card objects are immutable (dataclass with value/suit), so they don't need copying.
        This is significantly faster than copy.deepcopy() for MCTS simulations.
        
        Returns:
            A new GameState instance with copied mutable collections
        """
        return GameState(
            dungeon=self.dungeon.copy(),  # Shallow copy of list (Cards are immutable)
            room=self.room.copy(),
            discard=self.discard.copy(),
            equipped_weapon=self.equipped_weapon,  # Card is immutable, reference is fine
            weapon_monsters=self.weapon_monsters.copy(),
            used_potion=self.used_potion,
            health=self.health,
            number_avoided=self.number_avoided,
            last_room_avoided=self.last_room_avoided,
            exit=self.exit
        )
