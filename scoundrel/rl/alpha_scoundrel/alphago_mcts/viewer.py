"""
Interactive viewer for watching AlphaGo-style MCTS agent play Scoundrel.
"""
import argparse
import os
import multiprocessing as mp

# Set multiprocessing start method to 'spawn' for PyTorch compatibility
# This prevents CUDA/MPS context issues with fork on macOS
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

from scoundrel.models.game_state import Action, GameState
from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_agent import AlphaGoAgent
from scoundrel.rl.alpha_scoundrel.alphago_mcts.constants import (
    ALPHAGO_NUM_SIMULATIONS,
    ALPHAGO_NUM_WORKERS,
    ALPHAGO_VALUE_WEIGHT,
)
from scoundrel.rl.utils import format_action, denormalize_score
from scoundrel.rl.viewer import run_interactive_viewer


def _format_alphago_ui_text(action_enum: Action, extra_info: tuple) -> str:
    """
    Format UI text showing action statistics with policy priors.
    
    Display format:
    Next: [use 2] | [1:+15/200/0.15] [2:+18/350/0.40] [3:-5/100/0.10] [4:-8/80/0.05] [A:+2/70/0.30]
    
    Shows:
    - Selected action (bold green)
    - All actions with: score/visits/prior_prob
    
    Args:
        action_enum: Selected action
        extra_info: Tuple of (action_idx, stats_dict)
    
    Returns:
        Formatted UI text string
    """
    action_idx, stats = extra_info
    action_text = format_action(action_enum)
    ui_text = f"Next: [bold green]{action_text}[/bold green]"
    
    if stats:
        score_parts = []
        
        # Sort stats by action index for consistent display
        sorted_actions = sorted(stats.keys())
        
        for action in sorted_actions:
            s = stats[action]
            
            # Label: 1-4 for cards, A for avoid
            if action == 4:
                label = "A"
            else:
                label = str(action + 1)
            
            visits = s['visits']
            raw_score = denormalize_score(s['mean_value'])
            prior_prob = s.get('prior_prob', 0.0)
            
            # Format: [label:score/visits/prior]
            if action == action_idx:
                # Highlight selected action
                if raw_score > 0:
                    score_parts.append(f"[bold green][{label}:{raw_score:+d}/{visits}/{prior_prob:.2f}][/bold green]")
                else:
                    score_parts.append(f"[bold yellow][{label}:{raw_score:+d}/{visits}/{prior_prob:.2f}][/bold yellow]")
            else:
                # Non-selected actions
                if raw_score > 0:
                    score_parts.append(f"[green]{label}:{raw_score:+d}/{visits}/{prior_prob:.2f}[/green]")
                else:
                    score_parts.append(f"[dim]{label}:{raw_score:+d}/{visits}/{prior_prob:.2f}[/dim]")
        
        ui_text += " | " + " ".join(score_parts)
    
    return ui_text


def run_alphago_viewer(
    policy_large_checkpoint: str = None,
    policy_small_checkpoint: str = None,
    value_checkpoint: str = None,
    num_simulations: int = ALPHAGO_NUM_SIMULATIONS,
    num_workers: int = ALPHAGO_NUM_WORKERS,
    value_weight: float = ALPHAGO_VALUE_WEIGHT,
    seed: int = None
):
    """
    Run interactive viewer with AlphaGo MCTS agent.
    
    Args:
        policy_large_checkpoint: Path to policy network (default: from constants)
        policy_small_checkpoint: Path to fast rollout policy (default: from constants)
        value_checkpoint: Path to value network (default: from constants)
        num_simulations: Number of simulations per move
        num_workers: Number of parallel workers
        value_weight: λ for mixing value net and rollout (0-1)
        seed: Optional seed for deterministic games
    """
    os.system('resize -s 14 88')
    
    try:
        agent = AlphaGoAgent(
            policy_large_checkpoint=policy_large_checkpoint,
            policy_small_checkpoint=policy_small_checkpoint,
            value_checkpoint=value_checkpoint,
            num_simulations=num_simulations,
            num_workers=num_workers,
            value_weight=value_weight,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize AlphaGo agent: {e}")
        print("Please check that model checkpoints are available and valid.")
        print(f"\nCheckpoint paths:")
        print(f"  Policy Large: {policy_large_checkpoint or 'default'}")
        print(f"  Policy Small: {policy_small_checkpoint or 'default'}")
        print(f"  Value Large: {value_checkpoint or 'default'}")
        return
    
    parallel_str = f" | {num_workers} workers" if num_workers > 1 else ""
    label = f"AlphaGo MCTS ({num_simulations} sims, λ={value_weight}{parallel_str})"
    
    def get_alphago_action(state: GameState) -> tuple:
        """Get AlphaGo MCTS action and return with stats."""
        # Game should not be over when agent is asked for action
        if state.game_over:
            raise ValueError("Cannot get action when game is over")

        try:
            action_idx = agent.select_action(state)
            action_enum = agent.translator.decode_action(action_idx)
            stats = agent.get_action_stats()
            return action_enum, (action_idx, stats)
        except Exception as e:
            # If action selection fails, fall back to a simple heuristic
            print(f"\nWarning: Action selection failed: {e}")
            print("Falling back to first valid action...")
            # Just take first card
            from scoundrel.models.game_state import Action
            return Action.USE_1, (0, {})
    
    try:
        run_interactive_viewer(
            get_action_fn=get_alphago_action,
            label=label,
            seed=seed,
            format_ui_text_fn=_format_alphago_ui_text
        )
    except KeyboardInterrupt:
        print("\n\nViewer interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nERROR: Viewer crashed: {e}")
        print("This may be due to a multiprocessing or GPU issue.")
        import traceback
        traceback.print_exc()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Watch AlphaGo-style MCTS agent play Scoundrel interactively."
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=ALPHAGO_NUM_SIMULATIONS,
        help=f"Number of MCTS simulations per move (default: {ALPHAGO_NUM_SIMULATIONS})"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=ALPHAGO_NUM_WORKERS,
        help=f"Number of parallel workers (default: {ALPHAGO_NUM_WORKERS})"
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=ALPHAGO_VALUE_WEIGHT,
        help=f"λ for mixing value net and rollout, 0-1 (default: {ALPHAGO_VALUE_WEIGHT})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic deck shuffling (same seed = same game sequence)"
    )
    parser.add_argument(
        "--policy-large",
        type=str,
        default=None,
        help="Path to PolicyLarge checkpoint (default: from constants.py)"
    )
    parser.add_argument(
        "--policy-small",
        type=str,
        default=None,
        help="Path to PolicySmall checkpoint (default: from constants.py)"
    )
    parser.add_argument(
        "--value",
        type=str,
        default=None,
        help="Path to ValueLarge checkpoint (default: from constants.py)"
    )
    return parser.parse_args()


def main():
    """Entry point for console script."""
    args = parse_args()
    run_alphago_viewer(
        policy_large_checkpoint=args.policy_large,
        policy_small_checkpoint=args.policy_small,
        value_checkpoint=args.value,
        num_simulations=args.num_simulations,
        num_workers=args.num_workers,
        value_weight=args.value_weight,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

