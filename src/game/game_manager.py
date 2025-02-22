from typing import List, Optional
from ..models.game_state import GameState
from ..models.card import Card, CardType
from ..game.deck import Deck
from ..game.combat import Combat
from ..ui.terminal_ui import TerminalUI


class GameManager:
    def __init__(self):
        self.state = GameState()
        self.ui = TerminalUI()
        self.setup_game()

    def setup_game(self):
        self.state.dungeon = Deck.create_deck()
        self.draw_room()

    def draw_room(self):
        cards_needed = 4 - len(self.state.room)
        for _ in range(cards_needed):
            if self.state.dungeon:
                self.state.room.append(self.state.dungeon.pop())

    def handle_card(self, card: Card):
        if card.type == CardType.WEAPON:
            if self.state.equipped_weapon:
                self.state.discard.extend(
                    [self.state.equipped_weapon] + self.state.weapon_monsters
                )
                self.state.weapon_monsters = []
            self.state.equipped_weapon = card

        elif card.type == CardType.POTION:
            if not any(
                c.type == CardType.POTION for c in self.state.discard
            ):  # First potion this turn
                self.state.health = min(20, self.state.health + card.value)
            self.state.discard.append(card)

        elif card.type == CardType.MONSTER:
            if Combat.can_use_weapon(self.state, card):
                damage = Combat.calculate_damage(card, self.state.equipped_weapon)
                self.state.weapon_monsters.append(card)
            else:
                damage = card.value
                self.state.discard.append(card)
            self.state.health -= damage

    def avoid_room(self):
        if self.state.last_room_avoided:
            return False
        self.state.dungeon.extend(self.state.room)
        self.state.room = []
        self.state.last_room_avoided = True
        return True

    def play_turn(self):
        self.ui.display_game_state(self.state)

        if len(self.state.room) < 4:
            self.draw_room()

        # Get player choice
        print("\nOptions:")
        print("0. Avoid room")
        for i, card in enumerate(self.state.room, 1):
            print(f"{i}. Take {card}")

        while True:
            try:
                choice = int(input("\nEnter your choice (0-4): "))
                if choice == 0:
                    if self.avoid_room():
                        break
                    else:
                        print("Cannot avoid two rooms in a row!")
                elif 1 <= choice <= len(self.state.room):
                    card = self.state.room.pop(choice - 1)
                    self.handle_card(card)
                    self.state.last_room_avoided = False
                    break
            except ValueError:
                print("Invalid input!")

        return not self.state.game_over
