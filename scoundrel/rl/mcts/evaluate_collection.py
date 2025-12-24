"""
Console script for evaluating collected MCTS game data.

Reads all collected game JSON files and calculates summary statistics.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def get_default_collection_dir() -> Path:
    """Get the default directory for collected games."""
    script_dir = Path(__file__).parent
    return script_dir / "logs" / "collected_games"


def calculate_statistics(scores: List[int], total_turns: int, num_games: int) -> Dict[str, Any]:
    """
    Calculate summary statistics from game scores and turns.
    
    Args:
        scores: List of final scores from games
        total_turns: Total number of turns across all games
        num_games: Number of games
        
    Returns:
        Dictionary with calculated statistics
    """
    wins = sum(1 for s in scores if s > 0)
    win_percentage = (wins / num_games) * 100.0 if num_games > 0 else 0.0
    average_score = sum(scores) / num_games if num_games > 0 else 0.0
    best_score = max(scores) if scores else 0
    worst_score = min(scores) if scores else 0
    avg_turns = total_turns / num_games if num_games > 0 else 0.0
    
    return {
        "num_games": num_games,
        "wins": wins,
        "win_percentage": win_percentage,
        "average_score": average_score,
        "best_score": best_score,
        "worst_score": worst_score,
        "avg_turns": avg_turns,
        "total_turns": total_turns,
    }


def print_statistics(
    results: Dict[str, Any],
    title: str = "Collection Statistics",
    output_dir: Optional[str] = None,
    failed_games: int = 0,
) -> None:
    """
    Print summary statistics in a formatted way.
    
    Args:
        results: Dictionary with statistics (from calculate_statistics)
        title: Title to print before statistics
        output_dir: Optional output directory to display
        failed_games: Number of failed games (optional)
    """
    print()
    print(f"=== {title} ===")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print(f"Games: {results['num_games']}")
    if failed_games > 0:
        print(f"Failed to load: {failed_games}")
    print()
    print("Statistics:")
    print(f"  Wins: {results['wins']} ({results['win_percentage']:.2f}%)")
    print(f"  Average score: {results['average_score']:.2f}")
    print(f"  Best score: {results['best_score']}")
    print(f"  Worst score: {results['worst_score']}")
    print(f"  Average turns per game: {results['avg_turns']:.1f}")
    print(f"  Total turns: {results['total_turns']}")


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


def create_bar_graphs(
    scores: List[int],
    game_lengths: List[int],
    output_dir: Path,
) -> None:
    """
    Create frequency distribution bar charts for game scores and game lengths.
    
    Args:
        scores: List of game scores
        game_lengths: List of game lengths (number of turns)
        output_dir: Directory to save the graphs
    """
    if not scores or not game_lengths:
        print("Warning: No data to plot")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fig.suptitle('MCTS Collection Evaluation - Frequency Distributions', fontsize=16, fontweight='bold')
    
    # Game Scores frequency distribution
    # Create bins for scores
    min_score = min(scores)
    max_score = max(scores)
    # Use integer bins, ensuring we cover the full range
    bins_scores = np.arange(min_score - 0.5, max_score + 1.5, 1)
    counts_scores, bins_scores = np.histogram(scores, bins=bins_scores)
    bin_centers_scores = (bins_scores[:-1] + bins_scores[1:]) / 2
    
    ax1.bar(bin_centers_scores, counts_scores, width=0.8, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Game Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Game Score Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axvline(x=0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add average line
    avg_score = np.mean(scores)
    ax1.axvline(x=avg_score, color='green', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Average: {avg_score:.1f}')
    ax1.legend()
    
    # Game Lengths frequency distribution
    # Create bins for game lengths
    min_length = min(game_lengths)
    max_length = max(game_lengths)
    # Use integer bins
    bins_lengths = np.arange(min_length - 0.5, max_length + 1.5, 1)
    counts_lengths, bins_lengths = np.histogram(game_lengths, bins=bins_lengths)
    bin_centers_lengths = (bins_lengths[:-1] + bins_lengths[1:]) / 2
    
    ax2.bar(bin_centers_lengths, counts_lengths, width=0.8, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Game Length (Turns)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Game Length Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add average line
    avg_length = np.mean(game_lengths)
    ax2.axvline(x=avg_length, color='green', linestyle='--', linewidth=2,
                alpha=0.7, label=f'Average: {avg_length:.1f}')
    ax2.legend()
    
    plt.tight_layout()
    
    # Display the figure (persists until user closes)
    plt.show()


def evaluate_collection(collection_dir: Optional[Path] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate all collected games in a directory.
    
    Args:
        collection_dir: Directory containing collected game JSON files (default: mcts/logs/collected_games)
        verbose: Print detailed progress
        
    Returns:
        Dictionary with evaluation statistics
    """
    # Set up collection directory
    if collection_dir is None:
        collection_dir = get_default_collection_dir()
    
    collection_dir = Path(collection_dir)
    
    if not collection_dir.exists():
        print(f"Error: Collection directory does not exist: {collection_dir}")
        return {
            "num_games": 0,
            "error": f"Directory not found: {collection_dir}"
        }
    
    if verbose:
        print(f"Evaluating games in: {collection_dir}")
    
    # Find all JSON files
    json_files = sorted(collection_dir.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in {collection_dir}")
        return {
            "num_games": 0,
            "error": "No games found"
        }
    
    if verbose:
        print(f"Found {len(json_files)} game files")
        print()
    
    # Load and process all games
    scores = []
    game_lengths = []
    total_turns = 0
    loaded_games = 0
    failed_games = 0
    
    for filepath in json_files:
        if verbose:
            print(f"Loading {filepath.name}...", end=" ", flush=True)
        
        game_data = load_game_data(filepath)
        
        if game_data is None:
            failed_games += 1
            if verbose:
                print("FAILED")
            continue
        
        # Extract statistics
        metadata = game_data.get("metadata", {})
        score = metadata.get("final_score", 0)
        num_turns = metadata.get("num_turns", 0)
        
        scores.append(score)
        game_lengths.append(num_turns)
        total_turns += num_turns
        loaded_games += 1
        
        if verbose:
            print(f"Score={score}, Turns={num_turns}")
    
    if loaded_games == 0:
        print("Error: No valid games could be loaded")
        return {
            "num_games": 0,
            "error": "No valid games loaded"
        }
    
    # Calculate statistics
    results = calculate_statistics(scores, total_turns, loaded_games)
    results["failed_games"] = failed_games
    results["collection_dir"] = str(collection_dir)
    
    # Create bar graphs
    create_bar_graphs(scores, game_lengths, collection_dir)
    
    # Print results
    print_statistics(
        results,
        title="Collection Evaluation",
        output_dir=str(collection_dir),
        failed_games=failed_games,
    )
    
    return results


def main():
    """Console script entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate collected MCTS game data and print summary statistics."
    )
    parser.add_argument(
        "--collection-dir",
        type=str,
        default=None,
        help="Directory containing collected game JSON files (default: mcts/logs/collected_games)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress during evaluation"
    )
    
    args = parser.parse_args()
    
    collection_dir = Path(args.collection_dir) if args.collection_dir else None
    
    evaluate_collection(
        collection_dir=collection_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
