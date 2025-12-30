"""
Console script for evaluating and comparing collected game data.

Compares two directories of collected games (e.g., AlphaGo MCTS vs vanilla MCTS)
and provides comprehensive statistical analysis and visualizations.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_game_data(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Load game data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Game data dictionary, or None if file is invalid
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load {filepath}: {e}")
        return None


def calculate_statistics(scores: List[int], game_lengths: List[int]) -> Dict[str, Any]:
    """
    Calculate summary statistics from game scores and lengths.
    
    Args:
        scores: List of final scores from games
        game_lengths: List of game lengths (number of turns)
        
    Returns:
        Dictionary with calculated statistics
    """
    num_games = len(scores)
    wins = sum(1 for s in scores if s > 0)
    win_percentage = (wins / num_games) * 100.0 if num_games > 0 else 0.0
    average_score = sum(scores) / num_games if num_games > 0 else 0.0
    best_score = max(scores) if scores else 0
    worst_score = min(scores) if scores else 0
    avg_turns = sum(game_lengths) / num_games if num_games > 0 else 0.0
    total_turns = sum(game_lengths)
    
    # Calculate standard deviations
    score_std = np.std(scores) if scores else 0.0
    turns_std = np.std(game_lengths) if game_lengths else 0.0
    
    return {
        "num_games": num_games,
        "wins": wins,
        "win_percentage": win_percentage,
        "average_score": average_score,
        "score_std": score_std,
        "best_score": best_score,
        "worst_score": worst_score,
        "avg_turns": avg_turns,
        "turns_std": turns_std,
        "total_turns": total_turns,
        "scores": scores,
        "game_lengths": game_lengths,
    }


def print_statistics(
    results: Dict[str, Any],
    title: str = "Statistics",
    agent_type: Optional[str] = None,
) -> None:
    """
    Print summary statistics in a formatted way.
    
    Args:
        results: Dictionary with statistics
        title: Title to print before statistics
        agent_type: Type of agent (for display)
    """
    print()
    print(f"=== {title} ===")
    if agent_type:
        print(f"Agent Type: {agent_type}")
    print(f"Games: {results['num_games']}")
    print()
    print("Statistics:")
    print(f"  Wins: {results['wins']} ({results['win_percentage']:.2f}%)")
    print(f"  Average score: {results['average_score']:.2f} (±{results['score_std']:.2f})")
    print(f"  Best score: {results['best_score']}")
    print(f"  Worst score: {results['worst_score']}")
    print(f"  Average turns per game: {results['avg_turns']:.1f} (±{results['turns_std']:.1f})")
    print(f"  Total turns: {results['total_turns']}")


def load_collection(collection_dir: Path, verbose: bool = False) -> Tuple[List[int], List[int], str]:
    """
    Load all games from a collection directory.
    
    Args:
        collection_dir: Directory containing collected game JSON files
        verbose: Print detailed progress
        
    Returns:
        Tuple of (scores, game_lengths, agent_type)
    """
    if not collection_dir.exists():
        raise ValueError(f"Collection directory does not exist: {collection_dir}")
    
    json_files = sorted(collection_dir.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {collection_dir}")
    
    if verbose:
        print(f"Loading {len(json_files)} games from {collection_dir}")
    
    scores = []
    game_lengths = []
    agent_type = None
    failed_games = 0
    
    for filepath in json_files:
        game_data = load_game_data(filepath)
        
        if game_data is None:
            failed_games += 1
            continue
        
        # Extract statistics
        metadata = game_data.get("metadata", {})
        score = metadata.get("final_score", 0)
        num_turns = metadata.get("num_turns", 0)
        
        scores.append(score)
        game_lengths.append(num_turns)
        
        # Get agent type from first game
        if agent_type is None:
            agent_type = metadata.get("agent_type", "unknown")
    
    if failed_games > 0 and verbose:
        print(f"Warning: Failed to load {failed_games} games")
    
    if not scores:
        raise ValueError(f"No valid games loaded from {collection_dir}")
    
    return scores, game_lengths, agent_type


def compare_collections(
    dir1: Path,
    dir2: Path,
    label1: str = "Collection 1",
    label2: str = "Collection 2",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compare two collections of games.
    
    Args:
        dir1: First collection directory
        dir2: Second collection directory
        label1: Label for first collection
        label2: Label for second collection
        verbose: Print detailed progress
        
    Returns:
        Dictionary with comparison results
    """
    # Load both collections
    if verbose:
        print(f"\n=== Loading Collections ===")
    
    scores1, lengths1, agent_type1 = load_collection(dir1, verbose)
    scores2, lengths2, agent_type2 = load_collection(dir2, verbose)
    
    # Calculate statistics
    stats1 = calculate_statistics(scores1, lengths1)
    stats2 = calculate_statistics(scores2, lengths2)
    
    # Print individual statistics
    print_statistics(stats1, f"{label1} Statistics", agent_type1)
    print_statistics(stats2, f"{label2} Statistics", agent_type2)
    
    # Compare statistics
    print()
    print(f"=== Comparison: {label1} vs {label2} ===")
    print()
    
    # Win rate comparison
    win_rate_diff = stats1['win_percentage'] - stats2['win_percentage']
    print(f"Win Rate:")
    print(f"  {label1}: {stats1['win_percentage']:.2f}%")
    print(f"  {label2}: {stats2['win_percentage']:.2f}%")
    print(f"  Difference: {win_rate_diff:+.2f}%")
    print()
    
    # Average score comparison
    score_diff = stats1['average_score'] - stats2['average_score']
    score_diff_pct = (score_diff / stats2['average_score'] * 100) if stats2['average_score'] != 0 else 0
    print(f"Average Score:")
    print(f"  {label1}: {stats1['average_score']:.2f} (±{stats1['score_std']:.2f})")
    print(f"  {label2}: {stats2['average_score']:.2f} (±{stats2['score_std']:.2f})")
    print(f"  Difference: {score_diff:+.2f} ({score_diff_pct:+.1f}%)")
    print()
    
    # Game length comparison
    turns_diff = stats1['avg_turns'] - stats2['avg_turns']
    print(f"Average Game Length:")
    print(f"  {label1}: {stats1['avg_turns']:.1f} (±{stats1['turns_std']:.1f}) turns")
    print(f"  {label2}: {stats2['avg_turns']:.1f} (±{stats2['turns_std']:.1f}) turns")
    print(f"  Difference: {turns_diff:+.1f} turns")
    print()
    
    # Best/worst comparison
    print(f"Best Score:")
    print(f"  {label1}: {stats1['best_score']}")
    print(f"  {label2}: {stats2['best_score']}")
    print()
    
    print(f"Worst Score:")
    print(f"  {label1}: {stats1['worst_score']}")
    print(f"  {label2}: {stats2['worst_score']}")
    print()
    
    # Overall assessment
    improvements = []
    if win_rate_diff > 0:
        improvements.append(f"win rate (+{win_rate_diff:.2f}%)")
    if score_diff > 0:
        improvements.append(f"average score (+{score_diff:.2f})")
    
    if improvements:
        print(f"✓ {label1} outperforms {label2} in: {', '.join(improvements)}")
    elif win_rate_diff < 0 or score_diff < 0:
        print(f"✗ {label1} underperforms {label2}")
    else:
        print(f"≈ {label1} and {label2} perform similarly")
    
    return {
        "collection1": {
            "label": label1,
            "agent_type": agent_type1,
            "stats": stats1,
        },
        "collection2": {
            "label": label2,
            "agent_type": agent_type2,
            "stats": stats2,
        },
        "comparison": {
            "win_rate_diff": win_rate_diff,
            "score_diff": score_diff,
            "score_diff_pct": score_diff_pct,
            "turns_diff": turns_diff,
        }
    }


def create_comparison_plots(
    comparison_results: Dict[str, Any],
    show_plots: bool = True
) -> None:
    """
    Create side-by-side comparison plots.
    
    Args:
        comparison_results: Results from compare_collections
        show_plots: Whether to display plots
    """
    stats1 = comparison_results['collection1']['stats']
    stats2 = comparison_results['collection2']['stats']
    label1 = comparison_results['collection1']['label']
    label2 = comparison_results['collection2']['label']
    
    scores1 = stats1['scores']
    scores2 = stats2['scores']
    lengths1 = stats1['game_lengths']
    lengths2 = stats2['game_lengths']
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    fig.suptitle(f'Collection Comparison: {label1} vs {label2}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Score distributions
    min_score = min(min(scores1), min(scores2))
    max_score = max(max(scores1), max(scores2))
    bins_scores = np.arange(min_score - 0.5, max_score + 1.5, 1)
    
    ax1.hist(scores1, bins=bins_scores, alpha=0.5, label=label1, color='steelblue', edgecolor='black')
    ax1.hist(scores2, bins=bins_scores, alpha=0.5, label=label2, color='coral', edgecolor='black')
    ax1.axvline(x=stats1['average_score'], color='steelblue', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=stats2['average_score'], color='coral', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Game Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Score Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 2. Game length distributions
    min_length = min(min(lengths1), min(lengths2))
    max_length = max(max(lengths1), max(lengths2))
    bins_lengths = np.arange(min_length - 0.5, max_length + 1.5, 1)
    
    ax2.hist(lengths1, bins=bins_lengths, alpha=0.5, label=label1, color='steelblue', edgecolor='black')
    ax2.hist(lengths2, bins=bins_lengths, alpha=0.5, label=label2, color='coral', edgecolor='black')
    ax2.axvline(x=stats1['avg_turns'], color='steelblue', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(x=stats2['avg_turns'], color='coral', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Game Length (Turns)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Game Length Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Bar chart comparison of key metrics
    metrics = ['Win %', 'Avg Score', 'Avg Turns']
    values1 = [stats1['win_percentage'], stats1['average_score'], stats1['avg_turns']]
    values2 = [stats2['win_percentage'], stats2['average_score'], stats2['avg_turns']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize values for visual comparison
    normalized_values1 = [
        values1[0],  # Win % already in 0-100
        values1[1],  # Avg score
        values1[2]   # Avg turns
    ]
    normalized_values2 = [
        values2[0],
        values2[1],
        values2[2]
    ]
    
    ax3.bar(x - width/2, normalized_values1, width, label=label1, color='steelblue', alpha=0.7)
    ax3.bar(x + width/2, normalized_values2, width, label=label2, color='coral', alpha=0.7)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(normalized_values1, normalized_values2)):
        ax3.text(i - width/2, v1, f'{v1:.1f}', ha='center', va='bottom', fontsize=9)
        ax3.text(i + width/2, v2, f'{v2:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Box plots for score comparison
    ax4.boxplot([scores1, scores2], labels=[label1, label2], 
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if show_plots:
        plt.show()


def evaluate_single_collection(
    collection_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single collection directory.
    
    Args:
        collection_dir: Directory containing collected game JSON files
        verbose: Print detailed progress
        
    Returns:
        Dictionary with evaluation results
    """
    scores, lengths, agent_type = load_collection(collection_dir, verbose)
    stats = calculate_statistics(scores, lengths)
    
    print_statistics(stats, f"Evaluation: {collection_dir.name}", agent_type)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fig.suptitle(f'Collection Evaluation: {collection_dir.name}', 
                 fontsize=16, fontweight='bold')
    
    # Score distribution
    min_score = min(scores)
    max_score = max(scores)
    bins_scores = np.arange(min_score - 0.5, max_score + 1.5, 1)
    
    ax1.hist(scores, bins=bins_scores, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=stats['average_score'], color='green', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Average: {stats["average_score"]:.1f}')
    ax1.set_xlabel('Game Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Game length distribution
    min_length = min(lengths)
    max_length = max(lengths)
    bins_lengths = np.arange(min_length - 0.5, max_length + 1.5, 1)
    
    ax2.hist(lengths, bins=bins_lengths, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(x=stats['avg_turns'], color='green', linestyle='--', linewidth=2,
                alpha=0.7, label=f'Average: {stats["avg_turns"]:.1f}')
    ax2.set_xlabel('Game Length (Turns)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Game Length Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    return {
        "agent_type": agent_type,
        "stats": stats,
    }


def main():
    """Console script entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare collected game data."
    )
    parser.add_argument(
        "--dir1",
        type=str,
        required=True,
        help="First collection directory"
    )
    parser.add_argument(
        "--dir2",
        type=str,
        default=None,
        help="Second collection directory (for comparison)"
    )
    parser.add_argument(
        "--label1",
        type=str,
        default="Collection 1",
        help="Label for first collection"
    )
    parser.add_argument(
        "--label2",
        type=str,
        default="Collection 2",
        help="Label for second collection"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress during evaluation"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot visualization"
    )
    
    args = parser.parse_args()
    
    dir1 = Path(args.dir1)
    
    if args.dir2:
        # Compare two collections
        dir2 = Path(args.dir2)
        comparison_results = compare_collections(
            dir1=dir1,
            dir2=dir2,
            label1=args.label1,
            label2=args.label2,
            verbose=args.verbose
        )
        
        if not args.no_plots:
            create_comparison_plots(comparison_results, show_plots=True)
    else:
        # Evaluate single collection
        evaluate_single_collection(
            collection_dir=dir1,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()

