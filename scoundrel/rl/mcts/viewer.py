"""
Interactive viewer for watching MCTS agent play Scoundrel.
"""
import argparse
import os

from scoundrel.models.game_state import Action, GameState
from scoundrel.rl.mcts.mcts_agent import MCTSAgent
from scoundrel.rl.mcts.constants import MCTS_NUM_SIMULATIONS, MCTS_NUM_WORKERS
from scoundrel.rl.utils import format_action, denormalize_score
from scoundrel.rl.viewer import run_interactive_viewer


def _format_mcts_ui_text(action_enum: Action, extra_info: tuple[int, list]) -> str:
    """Format UI text for MCTS with action stats."""
    action_idx, stats = extra_info
    action_text = format_action(action_enum)
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
    
    return ui_text


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
    os.system('resize -s 14 88')
    agent = MCTSAgent(num_simulations=num_simulations, num_workers=num_workers)
    
    parallel_str = f" | {num_workers} workers" if num_workers > 1 else ""
    label = f"MCTS ({num_simulations} simulations{parallel_str})"
    
    def get_mcts_action(state: GameState) -> tuple[Action, tuple[int, list]]:
        """Get MCTS action and return with action_idx and stats."""
        action_idx = agent.select_action(state)
        action_enum = agent.translator.decode_action(action_idx)
        stats = agent.get_action_stats()
        return action_enum, (action_idx, stats)
    
    run_interactive_viewer(
        get_action_fn=get_mcts_action,
        label=label,
        seed=seed,
        format_ui_text_fn=_format_mcts_ui_text,
    )


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

