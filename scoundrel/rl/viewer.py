"""
Base viewer utilities for watching trained agents play Scoundrel.
"""
import time
from pathlib import Path
from typing import Callable, Any

from scoundrel.game.game_manager import GameManager
from scoundrel.models.game_state import Action, GameState
from scoundrel.rl.utils import format_action


def _autoplay_with_min_delay(
    engine: GameManager,
    get_action_fn: Callable[[GameState], tuple[Action, Any]],
    format_ui_text_fn: Callable[[Action, Any], str],
    actions_title: str,
    min_delay_seconds: float = 1.0,
) -> GameState:
    """
    Autoplay helper that executes actions with minimum delay.
    
    Measures time from action selection start to display end, then sleeps
    only the remaining time needed to reach minimum delay.
    
    Args:
        engine: GameManager instance
        get_action_fn: Function that takes state and returns (action, extra_info)
        format_ui_text_fn: Function that formats UI text from action and extra_info
        actions_title: Title for actions panel
        min_delay_seconds: Minimum time between action selections (default 1.0)
    
    Returns:
        Final game state after autoplay ends
    """
    state = engine.get_state()
    
    try:
        while not state.game_over and not state.exit:
            loop_start = time.time()
            
            # Get action (this may take time, especially for MCTS)
            try:
                action_enum, extra_info = get_action_fn(state)
            except (KeyboardInterrupt, EOFError):
                break
            
            # Execute action
            engine.execute_turn(action_enum)
            state = engine.get_state()
            
            if state.game_over:
                break
            
            # Format and display UI
            ui_text = format_ui_text_fn(action_enum, extra_info)
            engine.ui.display_game_state(
                state,
                actions_override=ui_text,
                actions_title=actions_title,
            )
            
            # Calculate elapsed time and sleep only if needed
            elapsed = time.time() - loop_start
            remaining_time = min_delay_seconds - elapsed
            if remaining_time > 0:
                try:
                    time.sleep(remaining_time)
                except (KeyboardInterrupt, EOFError):
                    # User interrupted during sleep
                    break
                
    except (KeyboardInterrupt, EOFError):
        # User interrupted autoplay
        pass
    
    return state


def run_interactive_viewer(
    get_action_fn: Callable[[Any], tuple[Action, Any]],
    label: str,
    checkpoint_name: str = "",
    seed: int = None,
    format_ui_text_fn: Callable[[Action, Any], str] = None,
):
    """
    Generic interactive viewer that can work with any agent type.
    
    Args:
        get_action_fn: Function that takes a game state and returns (action, extra_info)
        label: Label to display in the UI
        checkpoint_name: Optional checkpoint name to display
        seed: Optional seed for deterministic deck shuffling (same seed = same game sequence)
        format_ui_text_fn: Optional function to format UI text from (action, extra_info).
                          If None, uses default simple formatting.
    """
    engine = GameManager(seed=seed)
    
    actions_title = f"{label}"
    if checkpoint_name:
        actions_title += f" — {checkpoint_name}"
    
    # Default format function if none provided
    if format_ui_text_fn is None:
        def format_ui_text_fn(action: Action, extra_info: Any) -> str:
            action_text = format_action(action)
            return f"Next action: [bold]{action_text}[/bold]"
    
    state = engine.restart()
    
    try:
        while not state.exit:
            try:
                action_enum, extra_info = get_action_fn(state)
            except (KeyboardInterrupt, EOFError):
                engine.execute_turn(Action.EXIT)
                state = engine.get_state()
                break
            
            ui_text = format_ui_text_fn(action_enum, extra_info)
            
            if state.game_over:
                ui_text += " | Press 'R' to restart or 'E' to exit"
            
            engine.ui.display_game_state(
                state,
                actions_override=ui_text,
                actions_title=actions_title,
            )
            
            user = None
            try:
                user = input("Space=step | p=play | r=restart | q=quit: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                engine.execute_turn(Action.EXIT)
                state = engine.get_state()
                break
            
            if user in ("e", "exit", "q", "quit"):
                engine.execute_turn(Action.EXIT)
                state = engine.get_state()
                break
            if user in ("r", "restart"):
                state = engine.restart()
                continue
            if state.game_over:
                continue
            if user in ("p", "play"):
                state = _autoplay_with_min_delay(
                    engine,
                    get_action_fn,
                    format_ui_text_fn,
                    actions_title,
                )
                
                # Show final state after autoplay ends
                if state.game_over:
                    ui_text = f"Game Over | Press 'R' to restart or 'E' to exit"
                    engine.ui.display_game_state(
                        state,
                        actions_override=ui_text,
                        actions_title=actions_title,
                    )
                continue
            if user in ("", " ", "s", "step"):
                engine.execute_turn(action_enum)
                state = engine.get_state()
                continue
            state = engine.get_state()
    except (KeyboardInterrupt, EOFError):
        # Handle keyboard interrupt at top level
        engine.execute_turn(Action.EXIT)
        state = engine.get_state()
    
    engine.ui.display_game_state(state)



