"""
Interactive viewer for watching MCTS agent play Scoundrel.
"""
import argparse
from pathlib import Path

from scoundrel.game.game_manager import GameManager
from scoundrel.models.game_state import Action
from scoundrel.rl.mcts.mcts_agent import MCTSAgent
from scoundrel.rl.mcts.constants import MCTS_NUM_SIMULATIONS


def _format_action(action: Action) -> str:
    """Convert Action enum to human-readable string."""
    if action == Action.AVOID:
        return "avoid"
    if action in {Action.USE_1, Action.USE_2, Action.USE_3, Action.USE_4}:
        return f"use {action.value + 1}"
    return action.name.lower()


def _format_action_name(action_idx: int) -> str:
    """Convert action index to display name (1-4 for cards, 'avoid' for 4)."""
    if action_idx == 4:
        return "Avoid"
    return f"Card {action_idx + 1}"


def _denormalize_value(normalized_value: float) -> int:
    """Convert normalized value back to raw game score."""
    # Reverse of: (score + 188) / 218
    return int(normalized_value * 218 - 188)


def run_mcts_viewer(num_simulations: int = MCTS_NUM_SIMULATIONS):
    """
    Run interactive viewer with MCTS agent.
    
    Args:
        num_simulations: Number of MCTS simulations per move
    """
    agent = MCTSAgent(num_simulations=num_simulations)
    engine = GameManager()
    
    actions_title = f"MCTS ({num_simulations} simulations)"
    
    state = engine.restart()
    
    while not state.exit:
        # Get next action from MCTS
        action_idx = agent.select_action(state)
        action_enum = agent.translator.decode_action(action_idx)
        action_text = _format_action(action_enum)
        
        # Get action statistics
        stats = agent.get_action_stats(state)
        
        # Format compact action display: Next: [use 2] | Scores: 1:0.65/450 2:0.72/892 3:0.60/380 4:0.59/278
        ui_text = f"Next: [bold green]{action_text}[/bold green]"
        
        # Add compact scores with raw values
        if stats:
            score_parts = []
            for s in stats:
                # Use compact format: action_num:raw_score/visits
                if s['action'] == 4:
                    label = "A"  # Avoid
                else:
                    label = str(s['action'] + 1)
                
                visits = s['visits']
                raw_score = _denormalize_value(s['avg_value'])
                
                # Format: positive scores in green, negative in red/dim
                if s['action'] == action_idx:
                    # Selected action
                    if raw_score > 0:
                        score_parts.append(f"[bold green][{label}:{raw_score:+d}/{visits}][/bold green]")
                    else:
                        score_parts.append(f"[bold yellow][{label}:{raw_score:+d}/{visits}][/bold yellow]")
                else:
                    # Unselected action
                    if raw_score > 0:
                        score_parts.append(f"[green]{label}:{raw_score:+d}/{visits}[/green]")
                    else:
                        score_parts.append(f"[dim]{label}:{raw_score:+d}/{visits}[/dim]")
            
            ui_text += " | " + " ".join(score_parts)
        
        if state.game_over:
            ui_text += " | [r]estart [q]uit"
        
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
            # Ignore other inputs while game over
            continue
        if user in ("", " ", "s", "step"):
            engine.execute_turn(action_enum)
            state = engine.get_state()
            continue
        # Unrecognized input: loop and resample
        state = engine.get_state()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Watch MCTS agent play Scoundrel interactively."
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=MCTS_NUM_SIMULATIONS,
        help="Number of MCTS simulations per move"
    )
    return parser.parse_args()


def main():
    """Entry point for console script."""
    args = parse_args()
    run_mcts_viewer(num_simulations=args.num_simulations)


if __name__ == "__main__":
    main()

