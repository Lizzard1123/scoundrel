"""
Plot average score progression during a single MCTS episode.
Runs a greedy episode and displays the score over time.
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from scoundrel.game.game_manager import GameManager
from scoundrel.models.game_state import Action
from scoundrel.rl.mcts.mcts_agent import MCTSAgent
from scoundrel.rl.mcts.constants import MCTS_NUM_SIMULATIONS


def _denormalize_value(normalized_value: float) -> int:
    """Convert normalized value back to raw game score."""
    # Reverse of: (score + 188) / 218
    return int(normalized_value * 218 - 188)


def run_greedy_episode(num_simulations: int = MCTS_NUM_SIMULATIONS, verbose: bool = False):
    """
    Run a full episode using greedy MCTS and track scores at each step.
    
    Args:
        num_simulations: Number of MCTS simulations per move
        verbose: Whether to print step-by-step information
    
    Returns:
        tuple: (steps, scores) where steps is list of step numbers and scores is list of average scores
    """
    agent = MCTSAgent(num_simulations=num_simulations)
    engine = GameManager()
    
    state = engine.restart()
    
    steps = []
    scores = []
    step_count = 0
    
    if verbose:
        print(f"Starting episode with {num_simulations} simulations per move (greedy)")
        print(f"{'Step':<6} {'Action':<10} {'Avg Score':<12} {'Current Score':<14} {'Health':<8} {'Status'}")
        print("-" * 70)
    
    while not state.game_over:
        # Get next action from MCTS (greedy = most visited)
        action_idx = agent.select_action(state)
        action_enum = agent.translator.decode_action(action_idx)
        
        # Get action statistics to track average score
        stats = agent.get_action_stats(state)
        
        # Calculate weighted average score across all actions
        if stats:
            total_visits = sum(s['visits'] for s in stats)
            weighted_avg = sum(s['avg_value'] * s['visits'] for s in stats) / total_visits if total_visits > 0 else 0
            avg_raw_score = _denormalize_value(weighted_avg)
        else:
            avg_raw_score = 0
        
        # Track the data
        steps.append(step_count)
        scores.append(avg_raw_score)
        
        # Format action name for display
        if action_idx == 4:
            action_name = "Avoid"
        else:
            card = state.room[action_idx]
            action_name = f"{card.type.name} {card.value}"
        
        if verbose:
            print(f"{step_count:<6} {action_name:<10} {avg_raw_score:>+6d}      {state.score:>+6d}        {state.health:<8} {'OK' if state.health > 0 else 'DEAD'}")
        
        # Execute action
        engine.execute_turn(action_enum)
        state = engine.get_state()
        step_count += 1
    
    # Add final state
    steps.append(step_count)
    scores.append(state.score)  # Use actual final score
    
    if verbose:
        print("-" * 70)
        print(f"Episode finished!")
        print(f"Final Score: {state.score:+d}")
        print(f"Final Health: {state.health}")
        print(f"Total Steps: {step_count}")
        print()
    
    return steps, scores, state.score


def plot_episode(steps, scores, final_score, num_simulations):
    """
    Create and display a matplotlib line graph of the episode.
    
    Args:
        steps: List of step numbers
        scores: List of average scores at each step
        final_score: Final score achieved
        num_simulations: Number of simulations used
    """
    plt.figure(figsize=(12, 6))
    
    # Plot the line
    plt.plot(steps, scores, linewidth=2, color='#2E86AB', marker='o', markersize=4, alpha=0.8)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Styling
    plt.xlabel('Step', fontsize=12, fontweight='bold')
    plt.ylabel('Average Score', fontsize=12, fontweight='bold')
    plt.title(f'MCTS Episode Score Progression\n({num_simulations} simulations/move, Final Score: {final_score:+d})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add some styling
    plt.tight_layout()
    
    # Color the area under the curve
    plt.fill_between(steps, scores, 0, where=[s >= 0 for s in scores], alpha=0.2, color='green', interpolate=True)
    plt.fill_between(steps, scores, 0, where=[s < 0 for s in scores], alpha=0.2, color='red', interpolate=True)
    
    # Show the plot
    plt.show()


def run_batch_episodes(num_episodes: int, num_simulations: int = MCTS_NUM_SIMULATIONS, verbose: bool = False):
    """
    Run multiple episodes and collect score trajectories.
    
    Args:
        num_episodes: Number of episodes to run
        num_simulations: Number of MCTS simulations per move
        verbose: Whether to print episode summaries
    
    Returns:
        list: List of (steps, scores, final_score) tuples for each episode
    """
    episodes_data = []
    
    if verbose:
        print(f"Running {num_episodes} episodes with {num_simulations} simulations per move")
        print("=" * 70)
    
    for episode_num in range(num_episodes):
        if verbose:
            print(f"\nEpisode {episode_num + 1}/{num_episodes}:")
        
        steps, scores, final_score = run_greedy_episode(
            num_simulations=num_simulations,
            verbose=False
        )
        
        episodes_data.append((steps, scores, final_score))
        
        if verbose:
            print(f"  Final Score: {final_score:+d} | Steps: {len(steps)-1}")
    
    if verbose:
        print("\n" + "=" * 70)
        final_scores = [data[2] for data in episodes_data]
        print(f"Batch Summary:")
        print(f"  Mean Final Score: {np.mean(final_scores):+.1f}")
        print(f"  Std Final Score: {np.std(final_scores):.1f}")
        print(f"  Min Final Score: {min(final_scores):+d}")
        print(f"  Max Final Score: {max(final_scores):+d}")
        print()
    
    return episodes_data


def plot_batch_episodes(episodes_data, num_simulations, confidence_level=0.95):
    """
    Create and display a matplotlib line graph showing mean and confidence bounds.
    
    Args:
        episodes_data: List of (steps, scores, final_score) tuples
        num_simulations: Number of simulations used
        confidence_level: Confidence level for bounds (default 0.95 for 95%)
    """
    # Find the maximum number of steps across all episodes
    max_steps = max(len(steps) for steps, _, _ in episodes_data)
    
    # Create aligned arrays (pad shorter episodes with their final score)
    aligned_scores = []
    for steps, scores, _ in episodes_data:
        padded_scores = scores + [scores[-1]] * (max_steps - len(scores))
        aligned_scores.append(padded_scores)
    
    aligned_scores = np.array(aligned_scores)
    steps_array = np.arange(max_steps)
    
    # Calculate statistics
    mean_scores = np.mean(aligned_scores, axis=0)
    std_scores = np.std(aligned_scores, axis=0)
    
    # Calculate confidence bounds using standard error
    n_episodes = len(episodes_data)
    # Using t-distribution for small samples, z for large samples
    if n_episodes < 30:
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence_level) / 2, n_episodes - 1)
        margin = t_value * std_scores / np.sqrt(n_episodes)
    else:
        z_value = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        margin = z_value * std_scores / np.sqrt(n_episodes)
    
    lower_bound = mean_scores - margin
    upper_bound = mean_scores + margin
    
    # Create the plot
    plt.figure(figsize=(14, 7))
    
    # Plot individual episodes with low alpha
    for i, (steps, scores, _) in enumerate(episodes_data):
        plt.plot(steps, scores, linewidth=0.5, alpha=0.15, color='gray')
    
    # Plot confidence bounds
    plt.fill_between(steps_array, lower_bound, upper_bound, 
                     alpha=0.3, color='#2E86AB', label=f'{int(confidence_level*100)}% Confidence Interval')
    
    # Plot mean line
    plt.plot(steps_array, mean_scores, linewidth=2.5, color='#2E86AB', 
             label=f'Mean (n={n_episodes})', marker='o', markersize=3, markevery=max(1, max_steps//20))
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Styling
    plt.xlabel('Step', fontsize=12, fontweight='bold')
    plt.ylabel('Average Score', fontsize=12, fontweight='bold')
    
    final_scores = [data[2] for data in episodes_data]
    mean_final = np.mean(final_scores)
    std_final = np.std(final_scores)
    
    plt.title(f'MCTS Batch Episode Score Progression\n'
              f'({num_simulations} simulations/move, {n_episodes} episodes, '
              f'Final Score: {mean_final:+.1f} ± {std_final:.1f})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    plt.legend(loc='best', fontsize=10)
    
    # Add some styling
    plt.tight_layout()
    
    # Show the plot
    plt.show()



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a greedy MCTS episode and plot score progression."
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=MCTS_NUM_SIMULATIONS,
        help="Number of MCTS simulations per move (higher = more greedy)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Run multiple episodes and show mean with confidence bounds (e.g., --batch 10)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for batch mode bounds (default: 0.95)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print step-by-step information"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("MCTS Episode Plotter")
    print("=" * 70)
    print()
    
    if args.batch:
        # Batch mode: run multiple episodes and show mean with confidence bounds
        episodes_data = run_batch_episodes(
            num_episodes=args.batch,
            num_simulations=args.num_simulations,
            verbose=args.verbose or True  # Always show summary in batch mode
        )
        
        # Plot the batch results
        plot_batch_episodes(episodes_data, args.num_simulations, args.confidence)
    else:
        # Single episode mode
        steps, scores, final_score = run_greedy_episode(
            num_simulations=args.num_simulations,
            verbose=args.verbose
        )
        
        # Plot the results
        plot_episode(steps, scores, final_score, args.num_simulations)


if __name__ == "__main__":
    main()