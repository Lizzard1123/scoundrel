"""
Base viewer utilities for watching trained agents play Scoundrel.
"""
from pathlib import Path
from typing import Callable, Any

from scoundrel.game.game_manager import GameManager
from scoundrel.models.game_state import Action
from scoundrel.rl.utils import format_action


def run_interactive_viewer(
    get_action_fn: Callable[[Any], tuple[Action, Any]],
    label: str,
    checkpoint_name: str = "",
    seed: int = None,
):
    """
    Generic interactive viewer that can work with any agent type.
    
    Args:
        get_action_fn: Function that takes a game state and returns (action, extra_info)
        label: Label to display in the UI
        checkpoint_name: Optional checkpoint name to display
        seed: Optional seed for deterministic deck shuffling (same seed = same game sequence)
    """
    engine = GameManager(seed=seed)
    
    actions_title = f"{label}"
    if checkpoint_name:
        actions_title += f" — {checkpoint_name}"
    
    state = engine.restart()
    
    while not state.exit:
        action_enum, extra_info = get_action_fn(state)
        action_text = format_action(action_enum)
        ui_text = f"Next action: [bold]{action_text}[/bold]"
        
        if state.game_over:
            ui_text += " | press 'r' to restart or 'q' to quit"
        
        engine.ui.display_game_state(
            state,
            actions_override=ui_text,
            actions_title=actions_title,
        )
        
        user = input("Space=step | r=restart | q=quit: ").strip().lower()
        
        if user in ("q", "quit"):
            engine.execute_turn(Action.EXIT)
            state = engine.get_state()
            break
        if user in ("r", "restart"):
            state = engine.restart()
            continue
        if state.game_over:
            continue
        if user in ("", " ", "s", "step"):
            engine.execute_turn(action_enum)
            state = engine.get_state()
            continue
        state = engine.get_state()



