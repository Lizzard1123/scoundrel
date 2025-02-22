# scoundrel/game/game_manager.py
from typing import List, Optional, Tuple
from scoundrel.models.game_state import GameState
from scoundrel.models.card import Card, CardType
from scoundrel.game.deck import Deck
from scoundrel.game.combat import Combat
from scoundrel.ui.terminal_ui import TerminalUI


class GameManager:
    def __init__(self):
        self.state = GameState()
        self.ui = TerminalUI()
        self.setup_game()

    def setup_game(self):
        self.state.dungeon = Deck.create_deck()
        self.draw_room()

    def draw_room(self):
        # Only draw if there's 1 or 0 cards in the room
        if len(self.state.room) <= 1:
            cards_needed = 4 - len(self.state.room)
            for _ in range(cards_needed):
                if self.state.dungeon:
                    self.state.room.append(self.state.dungeon.pop())

    def parse_command(self, command: str) -> Tuple[str, int]:
        """Parse command string into action and card index"""
        parts = command.lower().strip().split()

        # Handle avoid command
        if parts[0] == "avoid" or parts[0] == "0":
            return "avoid", 0

        # Handle other commands
        if len(parts) == 2:
            action, index = parts
            try:
                index = int(index)
                if 1 <= index <= len(self.state.room):
                    return action, index
            except ValueError:
                pass
        elif len(parts) == 1:
            # Check if it's just a number
            try:
                index = int(parts[0])
                if 1 <= index <= len(self.state.room):
                    card = self.state.room[index - 1]
                    action = {
                        CardType.MONSTER: "fight",
                        CardType.WEAPON: "take",
                        CardType.POTION: "heal",
                    }[card.type]
                    return action, index
            except ValueError:
                pass

        return "invalid", 0

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

        if len(self.state.room) <= 1:
            self.draw_room()

        self.ui.display_game_state(self.state)

        while True:
            try:
                command = input("\nEnter command: ").strip()
                action, index = self.parse_command(command)

                if action == "invalid":
                    print(
                        "Invalid command! Use 'avoid' or '[fight/take/heal] [1-4]' or just the number"
                    )
                    continue

                if action == "avoid":
                    if self.avoid_room():
                        break
                    else:
                        print("Cannot avoid two rooms in a row!")
                else:
                    card = self.state.room[index - 1]
                    expected_action = {
                        CardType.MONSTER: "fight",
                        CardType.WEAPON: "take",
                        CardType.POTION: "heal",
                    }[card.type]

                    if action != expected_action:
                        print(f"Invalid action! Use '{expected_action}' for this card")
                        continue

                    self.state.room.pop(index - 1)
                    self.handle_card(card)
                    self.state.last_room_avoided = False
                    break

            except (ValueError, IndexError):
                print("Invalid input! Try again.")

        return not self.state.game_over
