"""
Console script for collecting MCTS game data for RL training.

Runs MCTS agent on games and logs comprehensive game data including:
- Game state snapshots at each turn
- MCTS statistics for all child nodes (actions)
- Final scores
"""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.mcts.mcts_agent import MCTSAgent
from scoundrel.models.game_state import GameState
from scoundrel.models.card import Card, Suit
from scoundrel.rl.mcts.constants import (
    MCTS_NUM_SIMULATIONS,
    MCTS_EXPLORATION_CONSTANT,
    MCTS_MAX_DEPTH,
    MCTS_NUM_WORKERS,
    USE_RANDOM_ROLLOUT,
)
from scoundrel.rl.mcts.evaluate_collection import calculate_statistics, print_statistics


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
    agent: MCTSAgent,
    engine: GameManager,
    game_seed: int,
    num_simulations: int,
) -> Dict[str, Any]:
    """
    Collect comprehensive data from a single game.
    
    Args:
        agent: MCTS agent to use
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
        
        # Run MCTS to select action
        action_idx = agent.select_action(state)
        
        # Get MCTS statistics for all child nodes
        mcts_stats = agent.get_action_stats()
        
        # Record event
        event = {
            "turn": turn,
            "game_state": serialize_game_state(game_state_snapshot),
            "mcts_stats": mcts_stats,
            "selected_action": action_idx,
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


def collect_games(
    num_games: int = None,
    seed: Optional[int] = None,
    num_simulations: int = MCTS_NUM_SIMULATIONS,
    output_dir: Path = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Collect data from multiple games using MCTS agent.
    
    Args:
        num_games: Number of games to collect (None = run until interrupted)
        seed: Base seed for GameManager (None = random seed)
        num_simulations: Number of MCTS simulations per move
        output_dir: Directory to save collected games (default: mcts/logs/collected_games)
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
        print(f"MCTS configuration:")
        print(f"  Simulations per move: {num_simulations}")
        print(f"  Exploration constant: {MCTS_EXPLORATION_CONSTANT}")
        print(f"  Max depth: {MCTS_MAX_DEPTH}")
        print(f"  Workers: {MCTS_NUM_WORKERS}")
        print(f"  Random rollout: {USE_RANDOM_ROLLOUT}")
        print()
    
    # Initialize agent
    agent = MCTSAgent(
        num_simulations=num_simulations,
        exploration_constant=MCTS_EXPLORATION_CONSTANT,
        max_depth=MCTS_MAX_DEPTH,
        num_workers=MCTS_NUM_WORKERS,
        use_random_rollout=USE_RANDOM_ROLLOUT,
    )
    
    saved_files = []
    scores = []
    total_turns = 0
    game_num = 0
    
    try:
        while num_games is None or game_num < num_games:
            game_num += 1
            game_seed = seed + game_num - 1
            
            if verbose:
                if num_games is None:
                    print(f"Collecting game {game_num} (seed={game_seed})...", end=" ", flush=True)
                else:
                    print(f"Collecting game {game_num}/{num_games} (seed={game_seed})...", end=" ", flush=True)
            
            # Create new engine for each game
            engine = GameManager(seed=game_seed)
            
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
    
    except KeyboardInterrupt:
        if verbose:
            print()
            print("\nInterrupted by user. Saving collected games...")
    
    # Calculate statistics using shared function
    num_collected = len(scores)
    results = calculate_statistics(scores, total_turns, num_collected)
    results["output_dir"] = str(output_dir)
    results["saved_files"] = saved_files
    
    # Print statistics using shared function
    if verbose:
        print_statistics(
            results,
            title="Collection Complete",
            output_dir=str(output_dir),
        )
    
    return results


def main():
    """Console script entry point."""
    parser = argparse.ArgumentParser(
        description="Collect MCTS game data for RL training."
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
        default=MCTS_NUM_SIMULATIONS,
        help=f"Number of MCTS simulations per move (default: {MCTS_NUM_SIMULATIONS})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save collected games (default: mcts/logs/collected_games)"
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
