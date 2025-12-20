"""
Pure game logic functions for Scoundrel.

This module contains stateless functions that implement game rules.
These functions can be used by both GameManager (for interactive play)
and MCTSAgent (for MCTS simulations) without code duplication.
"""
from scoundrel.models.game_state import GameState, Action
from scoundrel.models.card import Card, CardType
from scoundrel.game.combat import Combat


def apply_action_to_state(game_state: GameState, action: Action) -> GameState:
    """
    Pure function that applies an action to a game state and returns a new state.
    
    This is the core game logic - all game rules are implemented here.
    Used by both GameManager (for interactive play) and MCTSAgent (for MCTS).
    
    Args:
        game_state: Current game state (will be copied, not mutated)
        action: Action to apply
        
    Returns:
        New GameState with the action applied
    """
    new_state = game_state.copy()
    
    if new_state.game_over:
        return new_state
    
    match action:
        case Action.RESTART:
            new_state = GameState()
            return new_state
            
        case Action.EXIT:
            new_state.exit = True
            return new_state
            
        case Action.AVOID:
            if new_state.can_avoid:
                new_state.dungeon.extend(new_state.room)
                new_state.room = []
                new_state.number_avoided += 1
                new_state.last_room_avoided = True
                draw_room_in_state(new_state)
            return new_state
            
        case Action.USE_1 | Action.USE_2 | Action.USE_3 | Action.USE_4:
            if action.value >= len(new_state.room):
                return new_state
            
            card = new_state.room.pop(action.value)
            handle_card_in_state(new_state, card)
            
            if len(new_state.room) <= 1:
                draw_room_in_state(new_state)
            
            return new_state
            
        case _:
            return new_state


def handle_card_in_state(game_state: GameState, card: Card) -> None:
    """
    Handle a card being picked from the room.
    Mutates the game_state in place.
    
    Args:
        game_state: Game state to mutate
        card: Card that was picked
    """
    if card.type == CardType.WEAPON:
        if game_state.equipped_weapon:
            game_state.discard.extend(
                [game_state.equipped_weapon] + game_state.weapon_monsters
            )
            game_state.weapon_monsters = []
        game_state.equipped_weapon = card
        game_state.last_used_card = card
        
    elif card.type == CardType.POTION:
        if not game_state.used_potion:
            game_state.used_potion = True
            game_state.health = min(20, game_state.health + card.value)
        game_state.discard.append(card)
        game_state.last_used_card = card
        
    elif card.type == CardType.MONSTER:
        if Combat.can_use_weapon(game_state, card):
            damage = Combat.calculate_damage(card, game_state.equipped_weapon)
            game_state.weapon_monsters.append(card)
        else:
            damage = card.value
            game_state.discard.append(card)
        game_state.health -= damage
        game_state.last_used_card = card


def draw_room_in_state(game_state: GameState) -> None:
    """
    Draw cards from dungeon to fill room to 4 cards.
    Mutates the game_state in place.
    
    Args:
        game_state: Game state to mutate
    """
    if len(game_state.room) == 1:
        game_state.last_room_avoided = False
        game_state.used_potion = False
    cards_needed = 4 - len(game_state.room)
    for _ in range(cards_needed):
        if game_state.dungeon:
            game_state.room.append(game_state.dungeon.pop(0))
