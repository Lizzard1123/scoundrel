import random
from typing import List, Optional
from scoundrel.models.game_state import Action, GameState
from scoundrel.models.card import Card, CardAction, CardEffect, CardType
from scoundrel.game.deck import Deck
from scoundrel.game.combat import Combat
from scoundrel.game.game_logic import apply_action_to_state, handle_card_in_state, draw_room_in_state
from scoundrel.ui.terminal_ui import TerminalUI


class GameManager:
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the game manager.
        
        Args:
            seed: Optional seed for deterministic deck shuffling.
                  If provided, the same seed will produce the same deck order.
                  If None, a random seed is generated and used for reproducibility.
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        
        self.seed = seed
        self.state = GameState()
        self.command_text = ""
        self.ui = TerminalUI(seed=seed)
        self.setup_game()

    def setup_game(self):
        self.state.dungeon = Deck.create_deck(self.seed)
        self.draw_room()
        self.command_text = ""

    def restart(self) -> GameState:
        self.state = GameState()
        self.setup_game()
        return self.state

    def draw_room(self):
        """Draw cards from dungeon to fill room to 4 cards."""
        draw_room_in_state(self.state)

    def parse_command(self, command: str) -> Action:
        """Parse command string into action and card index"""
        parts = command.lower().strip().split()
        if(len(parts) == 0):
            parts = (None, None)
        if(len(parts) == 1):
            parts = (parts[0], None)
        if(len(parts) > 2):
            parts = (parts[0], parts[1])
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

    def get_state(self):
        return self.state

    def execute_turn(self, action: Action):
        """
        Execute a turn with the given action.
        Uses pure game logic function internally, then handles UI-specific concerns.
        """
        match action:
            case Action.RESTART:
                self.command_text = "Restarted"
                self.restart()
                return
            case Action.EXIT:
                self.command_text = "Exited"
                self.state.exit = True
                return
            case Action.AVOID:
                if self.state.game_over:
                    return
                if self.state.can_avoid:
                    self.command_text = "Avoided"
                else:
                    self.command_text = "Cannot avoid this room... Good Luck"
                    return  # Don't apply action if can't avoid
            case Action.USE_1 | Action.USE_2 | Action.USE_3 | Action.USE_4:
                if self.state.game_over:
                    return
                if action.value >= len(self.state.room):
                    self.command_text = f"Pick within 1-{len(self.state.room)}"
                    return
                card = self.state.room[action.value]
                log = CardEffect[card.type] + str(card)
                self.command_text = log
            case _:
                self.command_text = "Command not registered!"
                return
        
        new_state = apply_action_to_state(self.state, action)
        self.state = new_state

    def ui_loop(self):
        while not self.state.exit:
            self.ui.display_game_state(self.state)
            if self.state.game_over:
                self.command_text = "Press \'R\' to restart or \'E\' to exit"
            self.command_text += "\nEnter command: "
            command = None
            try:
                command = input(self.command_text).strip()
            except:
                self.execute_turn(Action.EXIT)
                continue
            action = self.parse_command(command)
            if action == Action.INVALID:
                self.command_text = "Invalid command! Use 'avoid' or '[fight/take/heal] [1-4]' or just the number"
                continue
            self.execute_turn(action)
        self.ui.display_game_state(self.state)
