"""
Interactive viewer for watching MCTS agent play Scoundrel.
"""
import argparse

from scoundrel.game.game_manager import GameManager
from scoundrel.models.game_state import Action
from scoundrel.rl.mcts.mcts_agent import MCTSAgent
from scoundrel.rl.mcts.constants import MCTS_NUM_SIMULATIONS, MCTS_NUM_WORKERS
from scoundrel.rl.utils import format_action, denormalize_score


def _format_action_name(action_idx: int) -> str:
    """Convert action index to display name (1-4 for cards, 'avoid' for 4)."""
    if action_idx == 4:
        return "Avoid"
    return f"Card {action_idx + 1}"


def run_mcts_viewer(
    num_simulations: int = MCTS_NUM_SIMULATIONS,
    num_workers: int = MCTS_NUM_WORKERS,
    seed: int = None,
):
    """
    Run interactive viewer with MCTS agent.
    
    Args:
        num_simulations: Number of MCTS simulations per move
        num_workers: Number of parallel workers (0 or 1 disables parallelization)
        seed: Optional seed for deterministic deck shuffling (same seed = same game sequence)
    """
    agent = MCTSAgent(num_simulations=num_simulations, num_workers=num_workers)
    engine = GameManager(seed=seed)
    
    parallel_str = f" | {num_workers} workers" if num_workers > 1 else ""
    actions_title = f"MCTS ({num_simulations} simulations{parallel_str})"
    
    state = engine.restart()
    
    while not state.exit:
        action_idx = agent.select_action(state)
        action_enum = agent.translator.decode_action(action_idx)
        action_text = format_action(action_enum)
        
        stats = agent.get_action_stats()
        
        ui_text = f"Next: [bold green]{action_text}[/bold green]"
        
        if stats:
            score_parts = []
            for s in stats:
                if s['action'] == 4:
                    label = "A"
                else:
                    label = str(s['action'] + 1)
                
                visits = s['visits']
                raw_score = denormalize_score(s['avg_value'])
                
                if s['action'] == action_idx:
                    if raw_score > 0:
                        score_parts.append(f"[bold green][{label}:{raw_score:+d}/{visits}][/bold green]")
                    else:
                        score_parts.append(f"[bold yellow][{label}:{raw_score:+d}/{visits}][/bold yellow]")
                else:
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
            continue
        if user in ("", " ", "s", "step"):
            engine.execute_turn(action_enum)
            state = engine.get_state()
            continue
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=MCTS_NUM_WORKERS,
        help="Number of parallel workers (0 or 1 disables parallelization)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic deck shuffling (same seed = same game sequence)"
    )
    return parser.parse_args()


def main():
    """Entry point for console script."""
    args = parse_args()
    run_mcts_viewer(
        num_simulations=args.num_simulations,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

