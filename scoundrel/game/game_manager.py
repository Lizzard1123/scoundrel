# scoundrel/game/game_manager.py
from typing import List, Optional, Tuple
from scoundrel.models.game_state import Action, GameState
from scoundrel.models.card import Card, CardAction, CardEffect, CardType
from scoundrel.game.deck import Deck
from scoundrel.game.combat import Combat
from scoundrel.ui.terminal_ui import TerminalUI


class GameManager:
    def __init__(self):
        self.state = GameState()
        self.command_text = ""
        self.ui = TerminalUI()
        self.setup_game()

    def setup_game(self):
        self.state.dungeon = Deck.create_deck()
        self.draw_room()
        self.command_text = ""

    def restart(self) -> GameState:
        self.state = GameState()
        self.setup_game()
        return self.state

    # Fresh room state
    def draw_room(self):
        if len(self.state.room) == 1:
            # Reset after full room completed
            self.state.last_room_avoided = False
            self.state.used_potion = False
        # Only draw if there's 1 or 0 cards in the room
        cards_needed = 4 - len(self.state.room)
        for _ in range(cards_needed):
            if self.state.dungeon:
                self.state.room.append(self.state.dungeon.pop(0))

    def avoid_room(self):
        self.state.dungeon.extend(self.state.room)
        self.state.room = []
        self.state.number_avoided += 1
        self.state.last_room_avoided = True

    def parse_command(self, command: str) -> Action:
        """Parse command string into action and card index"""
        parts = command.lower().strip().split()
        # Add padding
        if(len(parts) == 0):
            parts = (None, None)
        if(len(parts) == 1):
            parts = (parts[0], None)
        if(len(parts) > 2):
            parts = (parts[0], parts[1])
        # Match to action
        match parts:
            case ("avoid" | "0", _):
                return Action.AVOID
            case ("restart" | "r", _):
                return Action.RESTART
            case ("exit" | "e", _):
                return Action.EXIT
            case(index, None):
                try:
                    index = int(str(index))
                    return list(Action)[index-1]
                except:
                    return Action.INVALID
            case(action, index):
                try:
                    index = int(str(index))
                    card = self.state.room[index-1]
                    expected_action = str.lower(CardAction[card.type]).strip()
                    if action != expected_action:
                        return Action.INVALID
                    return list(Action)[index-1]
                except:
                    return Action.INVALID
            case _:
                return Action.INVALID

    def handle_card(self, card: Card):
        if card.type == CardType.WEAPON:
            if self.state.equipped_weapon:
                self.state.discard.extend(
                    [self.state.equipped_weapon] + self.state.weapon_monsters
                )
                self.state.weapon_monsters = []
            self.state.equipped_weapon = card

        elif card.type == CardType.POTION:
            if not self.state.used_potion: # First potion this turn
                self.state.used_potion = True
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

        return CardEffect[card.type] + str(card)

    def get_state(self):
        return self.state

    def execute_turn(self, action: Action):
        # Execute current action
        match action:
            case Action.RESTART:
                self.command_text = "Restarted"
                self.restart()
            case Action.EXIT:
                self.command_text = "Exited"
                self.state.exit = True
            case Action.AVOID:
                if(self.state.game_over):
                    return
                if self.state.can_avoid:
                    self.command_text = "Avoided"
                    self.avoid_room()
                else:
                    self.command_text = "Cannot avoid this room... Good Luck"
            case Action.USE_1 | Action.USE_2 | Action.USE_3 | Action.USE_4:
                if(self.state.game_over):
                    return
                if(action.value >= len(self.state.room)):
                    self.command_text = f"Pick within 1-{len(self.state.room)}"
                else:
                    card = self.state.room[action.value]
                    self.state.room.pop(action.value)
                    log = self.handle_card(card)
                    self.command_text = log
            case _:
                self.command_text = "Command not registered!"
        # Prepare for next turn
        if len(self.state.room) <= 1:
            self.draw_room()
        return

    def ui_loop(self):
        while not self.state.exit:
            # Display UI
            self.ui.display_game_state(self.state)
            # Display log and cmd input
            if self.state.game_over:
                self.command_text = "Press \'R\' to restart or \'E\' to exit"
            self.command_text += "\nEnter command: "
            command = None
            try:
                command = input(self.command_text).strip()
            except:
                self.execute_turn(Action.EXIT)
                continue
            # Parse input and execute
            action = self.parse_command(command)
            if action == Action.INVALID:
                self.command_text = "Invalid command! Use 'avoid' or '[fight/take/heal] [1-4]' or just the number"
                continue
            self.execute_turn(action)
        # Final game screen refresh
        self.ui.display_game_state(self.state)
