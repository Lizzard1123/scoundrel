"""
Console script for collecting AlphaGo MCTS game data for analysis.

Runs AlphaGo MCTS agent on games and logs comprehensive game data including:
- Game state snapshots at each turn
- MCTS statistics for all child nodes (actions)
- Final scores
"""
import argparse
import json
import random
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set multiprocessing start method to 'spawn' for PyTorch compatibility
# This prevents CUDA/MPS context issues with fork on macOS
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_agent import AlphaGoAgent
from scoundrel.models.game_state import GameState
from scoundrel.models.card import Card, Suit
from scoundrel.rl.alpha_scoundrel.alphago_mcts.constants import (
    ALPHAGO_NUM_SIMULATIONS,
    ALPHAGO_C_PUCT,
    ALPHAGO_VALUE_WEIGHT,
    ALPHAGO_MAX_DEPTH,
    ALPHAGO_NUM_WORKERS,
    POLICY_LARGE_CHECKPOINT,
    POLICY_SMALL_CHECKPOINT,
    VALUE_LARGE_CHECKPOINT,
)


def serialize_card(card: Card) -> Dict[str, Any]:
    """Serialize a Card to a dictionary."""
    return {
        "value": card.value,
        "suit": card.suit.value,  # Convert enum to string
    }


def serialize_game_state(game_state: GameState) -> Dict[str, Any]:
    """
    Serialize a GameState to a dictionary for JSON storage.
    
    Args:
        game_state: GameState to serialize
        
    Returns:
        Dictionary representation of the game state
    """
    return {
        "dungeon": [serialize_card(card) for card in game_state.dungeon],
        "room": [serialize_card(card) for card in game_state.room],
        "discard": [serialize_card(card) for card in game_state.discard],
        "equipped_weapon": serialize_card(game_state.equipped_weapon) if game_state.equipped_weapon else None,
        "weapon_monsters": [serialize_card(card) for card in game_state.weapon_monsters],
        "used_potion": game_state.used_potion,
        "health": game_state.health,
        "number_avoided": game_state.number_avoided,
        "last_room_avoided": game_state.last_room_avoided,
        "exit": game_state.exit,
        "score": game_state.score,
        "can_avoid": game_state.can_avoid,
        "can_use_potion": game_state.can_use_potion,
        "game_over": game_state.game_over,
    }


def collect_game_data(
    agent: AlphaGoAgent,
    engine: GameManager,
    game_seed: int,
    num_simulations: int,
) -> Dict[str, Any]:
    """
    Collect comprehensive data from a single game.
    
    Args:
        agent: AlphaGo MCTS agent to use
        engine: GameManager instance
        game_seed: Seed used for this game
        num_simulations: Number of MCTS simulations per move
        
    Returns:
        Dictionary containing game metadata and event sequence
    """
    events: List[Dict[str, Any]] = []
    state = engine.restart()
    turn = 0
    
    while not state.game_over:
        # Capture game state before action selection
        game_state_snapshot = state.copy()
        
        # Run AlphaGo MCTS to select action
        action_idx = agent.select_action(state)
        
        # Get MCTS statistics for all child nodes
        mcts_stats = agent.get_action_stats()
        
        # Get cache statistics
        cache_stats = agent.get_cache_stats()
        
        # Record event
        event = {
            "turn": turn,
            "game_state": serialize_game_state(game_state_snapshot),
            "mcts_stats": mcts_stats,
            "selected_action": action_idx,
            "cache_stats": cache_stats,
        }
        events.append(event)
        
        # Execute action
        action_enum = agent.translator.decode_action(action_idx)
        engine.execute_turn(action_enum)
        state = engine.get_state()
        
        turn += 1
    
    # Capture final state
    final_state = state.copy()
    final_score = state.score
    
    return {
        "metadata": {
            "seed": game_seed,
            "num_simulations": num_simulations,
            "timestamp": datetime.now().isoformat(),
            "final_score": final_score,
            "num_turns": turn,
            "final_state": serialize_game_state(final_state),
            "agent_type": "alphago_mcts",
            "config": {
                "c_puct": ALPHAGO_C_PUCT,
                "value_weight": ALPHAGO_VALUE_WEIGHT,
                "max_depth": ALPHAGO_MAX_DEPTH,
                "num_workers": ALPHAGO_NUM_WORKERS,
                "policy_large": POLICY_LARGE_CHECKPOINT,
                "policy_small": POLICY_SMALL_CHECKPOINT,
                "value_large": VALUE_LARGE_CHECKPOINT,
            }
        },
        "events": events,
    }


def save_game_data(game_data: Dict[str, Any], output_dir: Path, seed: int) -> Path:
    """
    Save game data to a JSON file.
    
    Args:
        game_data: Game data dictionary
        output_dir: Directory to save the file
        seed: Game seed (used as filename)
        
    Returns:
        Path to the saved file
    """
    filename = f"{seed}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(game_data, f, indent=2)
    
    return filepath


def calculate_statistics(scores: List[int], total_turns: int, num_collected: int) -> Dict[str, Any]:
    """
    Calculate summary statistics from game scores and turns.
    
    Args:
        scores: List of final scores from games
        total_turns: Total number of turns across all games
        num_collected: Number of games collected
        
    Returns:
        Dictionary with calculated statistics
    """
    wins = sum(1 for s in scores if s > 0)
    win_percentage = (wins / num_collected) * 100.0 if num_collected > 0 else 0.0
    average_score = sum(scores) / num_collected if num_collected > 0 else 0.0
    best_score = max(scores) if scores else 0
    worst_score = min(scores) if scores else 0
    avg_turns = total_turns / num_collected if num_collected > 0 else 0.0
    
    return {
        "num_games": num_collected,
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
) -> None:
    """
    Print summary statistics in a formatted way.
    
    Args:
        results: Dictionary with statistics (from calculate_statistics)
        title: Title to print before statistics
        output_dir: Optional output directory to display
    """
    print()
    print(f"=== {title} ===")
    if output_dir:
        print(f"Output directory: {output_dir}")
    print(f"Games: {results['num_games']}")
    print()
    print("Statistics:")
    print(f"  Wins: {results['wins']} ({results['win_percentage']:.2f}%)")
    print(f"  Average score: {results['average_score']:.2f}")
    print(f"  Best score: {results['best_score']}")
    print(f"  Worst score: {results['worst_score']}")
    print(f"  Average turns per game: {results['avg_turns']:.1f}")
    print(f"  Total turns: {results['total_turns']}")


def collect_games(
    num_games: int = None,
    seed: Optional[int] = None,
    num_simulations: int = ALPHAGO_NUM_SIMULATIONS,
    output_dir: Path = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Collect data from multiple games using AlphaGo MCTS agent.
    
    Args:
        num_games: Number of games to collect (None = run until interrupted)
        seed: Base seed for GameManager (None = random seed)
        num_simulations: Number of MCTS simulations per move
        output_dir: Directory to save collected games (default: alphago_mcts/logs/collected_games)
        verbose: Print progress during collection
        
    Returns:
        Dictionary with collection statistics
    """
    # Generate random seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
        if verbose:
            print(f"Using random seed: {seed}")
    
    # Set up output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "logs" / "collected_games"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Output directory: {output_dir}")
        print(f"Base seed: {seed}")
        if num_games is None:
            print("Collecting games (press Ctrl+C to stop)...")
        else:
            print(f"Collecting {num_games} games...")
        print(f"AlphaGo MCTS configuration:")
        print(f"  Simulations per move: {num_simulations}")
        print(f"  C_PUCT: {ALPHAGO_C_PUCT}")
        print(f"  Value Weight (λ): {ALPHAGO_VALUE_WEIGHT}")
        print(f"  Max depth: {ALPHAGO_MAX_DEPTH}")
        print(f"  Workers: {ALPHAGO_NUM_WORKERS}")
        print(f"  Policy Large: {POLICY_LARGE_CHECKPOINT}")
        print(f"  Policy Small: {POLICY_SMALL_CHECKPOINT}")
        print(f"  Value Large: {VALUE_LARGE_CHECKPOINT}")
        print()
    
    # Initialize agent (reuse same agent for all games)
    try:
        agent = AlphaGoAgent(
            num_simulations=num_simulations,
            c_puct=ALPHAGO_C_PUCT,
            value_weight=ALPHAGO_VALUE_WEIGHT,
            max_depth=ALPHAGO_MAX_DEPTH,
            num_workers=ALPHAGO_NUM_WORKERS,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize AlphaGo agent: {e}")
        print("Please check that model checkpoints are available and valid.")
        return {
            "num_games": 0,
            "wins": 0,
            "win_percentage": 0.0,
            "average_score": 0.0,
            "best_score": 0,
            "worst_score": 0,
            "avg_turns": 0.0,
            "total_turns": 0,
            "output_dir": str(output_dir),
            "saved_files": [],
            "failed_games": [],
            "num_failed": 0,
            "initialization_error": str(e),
        }
    
    saved_files = []
    scores = []
    total_turns = 0
    game_num = 0
    
    failed_games = []
    
    try:
        while num_games is None or game_num < num_games:
            game_num += 1
            game_seed = seed + game_num - 1
            
            try:
                if verbose:
                    if num_games is None:
                        print(f"Collecting game {game_num} (seed={game_seed})...", end=" ", flush=True)
                    else:
                        print(f"Collecting game {game_num}/{num_games} (seed={game_seed})...", end=" ", flush=True)
                
                # Create new engine for each game
                engine = GameManager(seed=game_seed)
                
                # Clear cache between games for consistent behavior
                agent.clear_cache()
                
                # Collect game data
                game_data = collect_game_data(agent, engine, game_seed, num_simulations)
                
                # Save game data
                filepath = save_game_data(
                    game_data,
                    output_dir,
                    game_seed
                )
                saved_files.append(str(filepath))
                
                score = game_data["metadata"]["final_score"]
                num_turns = game_data["metadata"]["num_turns"]
                scores.append(score)
                total_turns += num_turns
                
                if verbose:
                    print(f"Score={score}, Turns={num_turns}, Saved to {filepath.name}")
            
            except Exception as e:
                # Log the error but continue with next game
                failed_games.append((game_seed, str(e)))
                if verbose:
                    print(f"FAILED: {e}")
                    print(f"Continuing with next game...")
                # Continue to next game
                continue
    
    except KeyboardInterrupt:
        if verbose:
            print()
            print("\nInterrupted by user. Saving collected games...")
    
    # Calculate statistics
    num_collected = len(scores)
    results = calculate_statistics(scores, total_turns, num_collected)
    results["output_dir"] = str(output_dir)
    results["saved_files"] = saved_files
    results["failed_games"] = failed_games
    results["num_failed"] = len(failed_games)
    
    # Print statistics
    if verbose:
        print_statistics(
            results,
            title="Collection Complete",
            output_dir=str(output_dir),
        )
        
        # Print failed games summary if any
        if failed_games:
            print()
            print(f"=== Failed Games ({len(failed_games)}) ===")
            for seed, error in failed_games:
                print(f"  Seed {seed}: {error}")
    
    return results


def main():
    """Console script entry point."""
    parser = argparse.ArgumentParser(
        description="Collect AlphaGo MCTS game data for analysis."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help="Number of games to collect (default: None, run until interrupted)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for GameManager (default: random seed)"
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=ALPHAGO_NUM_SIMULATIONS,
        help=f"Number of MCTS simulations per move (default: {ALPHAGO_NUM_SIMULATIONS})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save collected games (default: alphago_mcts/logs/collected_games)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress during collection"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    collect_games(
        num_games=args.num_games,
        seed=args.seed,
        num_simulations=args.num_simulations,
        output_dir=output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

