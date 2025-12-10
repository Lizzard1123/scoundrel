from scoundrel.models.card import Card, CardType
from scoundrel.models.game_state import GameState


class Combat:
    @staticmethod
    def calculate_damage(monster: Card, weapon: Card | None = None) -> int:
        if not weapon:
            return monster.value

        damage = monster.value - weapon.value
        return max(0, damage)

    @staticmethod
    def can_use_weapon(game_state: GameState, monster: Card) -> bool:
        if not game_state.equipped_weapon:
            return False

        if not game_state.weapon_monsters:
            return True

        return monster.value <= game_state.weapon_monsters[-1].value
