"""
Game generation for AlphaGo self-play training.

Uses a single AlphaGoAgent with internal multithreading for MCTS,
matching the pattern used in eval.py for fast inference.
"""

import argparse
import json
import random
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_agent import AlphaGoAgent
from scoundrel.models.game_state import GameState
from scoundrel.models.card import Card, Suit
from scoundrel.rl.alpha_scoundrel.self_play.constants import (
    SELF_PLAY_NUM_WORKERS,
    SELF_PLAY_SIMULATIONS,
    SELF_PLAY_GAMES_PER_ITERATION,
    SELF_PLAY_PARALLEL_GAMES,
)

# Set multiprocessing start method to 'spawn' for PyTorch compatibility
# This prevents CUDA/MPS context issues with fork on macOS
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass


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


def collect_single_game_data(
    agent: AlphaGoAgent,
    engine: GameManager,
    game_seed: int,
    num_simulations: int,
) -> Dict[str, Any]:
    """
    Collect comprehensive data from a single self-play game.

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
            "agent_type": "alphago_self_play",
            "config": {
                "c_puct": agent.c_puct,
                "value_weight": agent.value_weight,
                "max_depth": agent.max_depth,
                "num_workers": agent.num_workers,
            }
        },
        "events": events,
    }


def print_statistics(
    results: Dict[str, Any],
    title: str = "Statistics",
    output_dir: Optional[str] = None,
) -> None:
    """
    Print summary statistics in a formatted way.

    Args:
        results: Dictionary with statistics
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


def game_generation_worker(
    worker_id: int,
    game_seed: int,
    num_simulations: int,
    policy_checkpoint: Optional[str],
    policy_small_checkpoint: Optional[str],
    value_checkpoint: Optional[str],
    output_dir: Path,
    progress_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """
    Worker function that generates a single self-play game and reports progress.

    Args:
        worker_id: Unique identifier for this worker
        game_seed: Seed for the game
        num_simulations: Number of MCTS simulations per move
        policy_checkpoint: Path to policy network checkpoint
        policy_small_checkpoint: Path to fast rollout policy checkpoint
        value_checkpoint: Path to value network checkpoint
        output_dir: Directory to save game data
        progress_queue: Queue for sending progress updates
        result_queue: Queue for sending final results
    """
    try:
        # Initialize agent in worker process (each process gets its own GPU context)
        agent = AlphaGoAgent(
            policy_large_checkpoint=policy_checkpoint,
            policy_small_checkpoint=policy_small_checkpoint,
            value_checkpoint=value_checkpoint,
            num_simulations=num_simulations,
            num_workers=SELF_PLAY_NUM_WORKERS,
            add_dirichlet_noise=True,  # Enable exploration noise for self-play
            game_seed=game_seed,  # Seed for reproducible policy sampling
        )

        # Create game engine
        engine = GameManager(seed=game_seed)

        # Send initial progress update
        progress_queue.put(('start', worker_id, game_seed, 0))

        # Initialize events list for RL training
        events: List[Dict[str, Any]] = []

        state = engine.restart()
        turn = 0

        while not state.game_over:
            # Send progress update for current turn
            progress_queue.put(('turn', worker_id, game_seed, turn))

            # Capture game state before action selection
            game_state_snapshot = state.copy()

            # Run AlphaGo MCTS to select action
            action_idx = agent.select_action(state)

            # Get MCTS statistics for all child nodes
            mcts_stats = agent.get_action_stats()

            # Get cache statistics
            cache_stats = agent.get_cache_stats()

            # Record event for RL training
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

        # Collect final game data
        final_state = state.copy()
        final_score = state.score

        game_data = {
            "metadata": {
                "seed": game_seed,
                "num_simulations": num_simulations,
                "timestamp": datetime.now().isoformat(),
                "final_score": final_score,
                "num_turns": turn,
                "final_state": serialize_game_state(final_state),
                "agent_type": "alphago_self_play",
                "config": {
                    "c_puct": agent.c_puct,
                    "value_weight": agent.value_weight,
                    "max_depth": agent.max_depth,
                    "num_workers": agent.num_workers,
                }
            },
            "events": events,
        }

        # Save game data
        filepath = save_game_data(game_data, output_dir, game_seed)

        # Send completion update
        progress_queue.put(('complete', worker_id, game_seed, turn))

        # Send result
        result_queue.put({
            'worker_id': worker_id,
            'game_seed': game_seed,
            'score': final_score,
            'turns': turn,
            'filepath': str(filepath),
            'success': True,
        })

    except Exception as e:
        # Send error update
        progress_queue.put(('error', worker_id, game_seed, str(e)))

        # Send error result
        result_queue.put({
            'worker_id': worker_id,
            'game_seed': game_seed,
            'error': str(e),
            'success': False,
        })


def generate_self_play_games_parallel(
    num_games: int,
    num_parallel_games: int,
    policy_checkpoint: Optional[Path],
    policy_small_checkpoint: Optional[Path],
    value_checkpoint: Optional[Path],
    output_dir: Path,
    num_simulations: int = SELF_PLAY_SIMULATIONS,
    base_seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generate self-play games using parallel processing with separate processes.

    This creates multiple AlphaGo agents in separate processes to avoid GPU/MPS
    contention issues, while still providing significant speedup over sequential generation.

    Args:
        num_games: Total number of games to generate
        num_parallel_games: Number of games to generate simultaneously
        policy_checkpoint: Path to policy network checkpoint
        policy_small_checkpoint: Path to fast rollout policy checkpoint
        value_checkpoint: Path to value network checkpoint
        output_dir: Directory to save generated games
        num_simulations: Number of MCTS simulations per move
        base_seed: Base seed for game generation (None = random)
        verbose: Print progress information with colored concurrent game status

    Returns:
        Dictionary with generation statistics
    """
    start_time = time.time()

    # Generate random seed if not provided
    if base_seed is None:
        base_seed = random.randint(0, 2**31 - 1)
        if verbose:
            print(f"Using random base seed: {base_seed}")

    # Set up output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Output directory: {output_dir}")
        print(f"Generating {num_games} self-play games with {num_parallel_games} parallel processes...")
        print(f"AlphaGo MCTS configuration:")
        print(f"  Simulations per move: {num_simulations}")
        print(f"  Internal MCTS workers: {SELF_PLAY_NUM_WORKERS}")
        print(f"  Policy checkpoint: {policy_checkpoint}")
        print(f"  Value checkpoint: {value_checkpoint}")
        print()

    # Initialize tracking variables
    saved_files = []
    scores = []
    total_turns = 0
    started_games = 0
    failed_games = []

    # Track active games for verbose display
    active_games = {}  # worker_id -> (game_seed, start_time, current_turn)
    completed_game_history = []  # List of (game_seed, score, turns, worker_id) tuples

    # Create queues for inter-process communication
    progress_queue = mp.Queue()
    result_queue = mp.Queue()

    # Track next game seed and worker assignments
    next_game_seed = base_seed
    available_workers = list(range(num_parallel_games))
    active_processes = {}  # worker_id -> process

    def start_new_game(worker_id: int) -> bool:
        """Start a new game for the given worker. Returns False if no more games to start."""
        nonlocal next_game_seed, started_games, num_games

        if started_games >= num_games:
            return False

        game_seed = next_game_seed
        next_game_seed += 1

        # Start process for this game
        process = mp.Process(
            target=game_generation_worker,
            args=(
                worker_id,
                game_seed,
                num_simulations,
                str(policy_checkpoint) if policy_checkpoint else None,
                str(policy_small_checkpoint) if policy_small_checkpoint else None,
                str(value_checkpoint) if value_checkpoint else None,
                output_dir,
                progress_queue,
                result_queue,
            )
        )
        process.start()
        active_processes[worker_id] = process

        # Increment the counter of games we've started
        started_games += 1

        return True

    def print_verbose_status():
        """Print the current status with history and active games - exact format from AlphaGo MCTS collection."""
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")  # Clear screen and move to top

        # Print header
        print("=== AlphaGo Self-Play Game Generation ===")
        print(f"Output: {output_dir}")
        print(f"Parallel games: {num_parallel_games} | Completed: {len(completed_game_history)}")
        if num_games is not None:
            print(f"Target: {num_games} games")
        print()

        # Print completed games history (last 10)
        if completed_game_history:
            print("=== Completed Games ===")
            for i, (game_seed, score, turns, worker_id) in enumerate(completed_game_history[-10:]):  # Show last 10
                status = "WIN" if score > 0 else "LOSE"
                color = "\033[92m" if score > 0 else "\033[91m"  # Green for win, red for loss
                print(f"{color}Game {game_seed}: {status} Score={score}, Turns={turns}, Worker={worker_id}\033[0m")
            if len(completed_game_history) > 10:
                print(f"... and {len(completed_game_history) - 10} more")
            print()

        # Print active games
        if active_games:
            print("=== Active Games ===")
            for worker_id, (game_seed, start_time, current_turn) in active_games.items():
                elapsed = time.time() - start_time
                color = "\033[94m"  # Blue for active
                print(f"{color}Worker {worker_id}: Game {game_seed}, Turn {current_turn}, {elapsed:.1f}s\033[0m")
            print()

    # Initialize verbose status tracking
    print_verbose_status.last_completed = 0

    try:
        # Start initial batch of games
        for worker_id in available_workers[:]:
            if not start_new_game(worker_id):
                available_workers.remove(worker_id)

        # Main generation loop
        while active_processes or started_games < num_games:
            # Check for progress updates
            while not progress_queue.empty():
                try:
                    msg_type, worker_id, game_seed, data = progress_queue.get_nowait()

                    if msg_type == 'start':
                        active_games[worker_id] = (game_seed, time.time(), 0)
                    elif msg_type == 'turn':
                        if worker_id in active_games:
                            _, start_time, _ = active_games[worker_id]
                            active_games[worker_id] = (game_seed, start_time, data)
                    elif msg_type == 'complete':
                        if worker_id in active_games:
                            del active_games[worker_id]
                    elif msg_type == 'error':
                        if worker_id in active_games:
                            del active_games[worker_id]
                except:
                    pass

            # Check for completed results
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()

                    if result['success']:
                        scores.append(result['score'])
                        total_turns += result['turns']
                        saved_files.append(result['filepath'])
                        completed_game_history.append((
                            result['game_seed'],
                            result['score'],
                            result['turns'],
                            result['worker_id']
                        ))
                    else:
                        failed_games.append((result['game_seed'], result.get('error', 'Unknown error')))

                    # Clean up finished process
                    if result['worker_id'] in active_processes:
                        active_processes[result['worker_id']].join()
                        del active_processes[result['worker_id']]

                    # Start new game for this worker if more games needed
                    # completed_games tracks started games, so we can start a new one
                    start_new_game(result['worker_id'])

                except:
                    pass

            # Update display if verbose
            if verbose:
                print_verbose_status()

            # Check if all processes are done and no more games to start
            if not active_processes and started_games >= num_games:
                break

            # Small delay to prevent busy waiting
            time.sleep(0.1)

    except KeyboardInterrupt:
        if verbose:
            print("\nInterrupted by user. Terminating processes...")
        # Terminate all active processes
        for process in active_processes.values():
            process.terminate()
        for process in active_processes.values():
            process.join()

    # Wait for any remaining processes
    for process in active_processes.values():
        process.join()

    # Calculate statistics
    num_collected = len(scores)
    wins = sum(1 for s in scores if s > 0)
    win_percentage = (wins / num_collected) * 100.0 if num_collected > 0 else 0.0
    average_score = sum(scores) / num_collected if num_collected > 0 else 0.0
    best_score = max(scores) if scores else 0
    worst_score = min(scores) if scores else 0
    avg_turns = total_turns / num_collected if num_collected > 0 else 0.0

    generation_time = time.time() - start_time
    games_per_hour = (num_collected / generation_time) * 3600 if generation_time > 0 else 0.0

    results = {
        "num_games": num_collected,
        "wins": wins,
        "win_percentage": win_percentage,
        "average_score": average_score,
        "best_score": best_score,
        "worst_score": worst_score,
        "avg_turns": avg_turns,
        "total_turns": total_turns,
        "games_per_hour": games_per_hour,
        "generation_time": generation_time,
        "output_dir": str(output_dir),
        "saved_files": saved_files,
        "failed_games": failed_games,
        "num_failed": len(failed_games),
        "base_seed": base_seed,
    }

    if verbose:
        print_verbose_status()  # Final status update
        print("=== Self-Play Generation Complete ===")
        print(f"Games Generated: {num_collected}")
        print(f"Wins: {wins} ({win_percentage:.2f}%)")
        print(f"Average Score: {average_score:.2f}")
        print(f"Best Score: {best_score}")
        print(f"Worst Score: {worst_score}")
        print(f"Average Turns: {avg_turns:.1f}")
        print(f"Games/Hour: {games_per_hour:.1f}")
        print(f"Generation Time: {generation_time:.1f}s")
        if failed_games:
            print(f"Failed Games: {len(failed_games)}")
        print(f"Output Directory: {output_dir}")
        print()

        # Print failed games summary if any
        if failed_games:
            print(f"=== Failed Games ({len(failed_games)}) ===")
            for seed, error in failed_games:
                print(f"  Seed {seed}: {error}")
            print()

    return results


def generate_self_play_games(
    num_games: int,
    num_workers: int,
    policy_checkpoint: Optional[Path],
    policy_small_checkpoint: Optional[Path],
    value_checkpoint: Optional[Path],
    output_dir: Path,
    num_simulations: int = SELF_PLAY_SIMULATIONS,
    num_parallel_games: int = SELF_PLAY_PARALLEL_GAMES,
    base_seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generate self-play games using AlphaGoAgent.

    This is a wrapper that uses parallel generation when SELF_PLAY_PARALLEL_GAMES > 1,
    otherwise falls back to sequential generation for compatibility.

    Args:
        num_games: Total number of games to generate
        num_workers: Number of internal MCTS workers (threading, not multiprocessing)
        policy_checkpoint: Path to policy network checkpoint
        policy_small_checkpoint: Path to fast rollout policy checkpoint
        value_checkpoint: Path to value network checkpoint
        output_dir: Directory to save generated games
        num_simulations: Number of MCTS simulations per move
        base_seed: Base seed for game generation (None = random)
        verbose: Print progress information

    Returns:
        Dictionary with generation statistics
    """
    # For single-threaded generation or when parallel games is 1, use sequential
    if num_parallel_games <= 1:
        if verbose:
            print(f"Using sequential generation (num_parallel_games={num_parallel_games})")
        return generate_self_play_games_sequential(
            num_games=num_games,
            num_workers=num_workers,
            policy_checkpoint=policy_checkpoint,
            policy_small_checkpoint=policy_small_checkpoint,
            value_checkpoint=value_checkpoint,
            output_dir=output_dir,
            num_simulations=num_simulations,
            base_seed=base_seed,
            verbose=verbose,
        )
    else:
        if verbose:
            print(f"Using parallel generation (num_parallel_games={num_parallel_games})")
        return generate_self_play_games_parallel(
            num_games=num_games,
            num_parallel_games=num_parallel_games,
            policy_checkpoint=policy_checkpoint,
            policy_small_checkpoint=policy_small_checkpoint,
            value_checkpoint=value_checkpoint,
            output_dir=output_dir,
            num_simulations=num_simulations,
            base_seed=base_seed,
            verbose=verbose,
        )


def generate_self_play_games_sequential(
    num_games: int,
    num_workers: int,
    policy_checkpoint: Optional[Path],
    policy_small_checkpoint: Optional[Path],
    value_checkpoint: Optional[Path],
    output_dir: Path,
    num_simulations: int = SELF_PLAY_SIMULATIONS,
    base_seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generate self-play games using a single AlphaGoAgent with internal multithreading.

    This matches the pattern used in eval.py - one agent with num_workers for
    internal root parallelization within MCTS. Much faster than multiprocessing
    because models are loaded once and stay on GPU.

    Args:
        num_games: Total number of games to generate
        num_workers: Number of internal MCTS workers (threading, not multiprocessing)
        policy_checkpoint: Path to policy network checkpoint
        policy_small_checkpoint: Path to fast rollout policy checkpoint
        value_checkpoint: Path to value network checkpoint
        output_dir: Directory to save generated games
        num_simulations: Number of MCTS simulations per move
        base_seed: Base seed for game generation (None = random)
        verbose: Print progress information

    Returns:
        Dictionary with generation statistics
    """
    start_time = time.time()

    # Generate random seed if not provided
    if base_seed is None:
        base_seed = random.randint(0, 2**31 - 1)
        if verbose:
            print(f"Using random base seed: {base_seed}")

    # Set up output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Output directory: {output_dir}")
        print(f"Generating {num_games} self-play games...")
        print(f"AlphaGo MCTS configuration:")
        print(f"  Simulations per move: {num_simulations}")
        print(f"  Internal MCTS workers: {num_workers}")
        print(f"  Policy checkpoint: {policy_checkpoint}")
        print(f"  Value checkpoint: {value_checkpoint}")
        print()

    # Create a SINGLE agent with internal multithreading (like eval.py)
    # Models loaded once, stay on GPU, use threading for MCTS parallelization
    try:
        agent = AlphaGoAgent(
            policy_large_checkpoint=str(policy_checkpoint) if policy_checkpoint else None,
            policy_small_checkpoint=str(policy_small_checkpoint) if policy_small_checkpoint else None,
            value_checkpoint=str(value_checkpoint) if value_checkpoint else None,
            num_simulations=num_simulations,
            num_workers=num_workers,  # Internal MCTS root parallelization
            add_dirichlet_noise=True, # Enable exploration noise for self-play
        )
        if verbose:
            print(f"Agent loaded on device: {agent.device}")
    except Exception as e:
        print(f"ERROR: Failed to initialize AlphaGo agent: {e}")
        return {
            "num_games": 0,
            "wins": 0,
            "win_percentage": 0.0,
            "average_score": 0.0,
            "output_dir": str(output_dir),
            "saved_files": [],
            "failed_games": [],
            "num_failed": 0,
            "error": str(e),
        }

    # Generate games sequentially (agent uses internal parallelization for speed)
    saved_files = []
    scores = []
    total_turns = 0
    failed_games = []

    try:
        for game_num in range(num_games):
            game_seed = base_seed + game_num

            try:
                if verbose:
                    print(f"Generating game {game_num + 1}/{num_games} (seed={game_seed})...", end=" ", flush=True)

                # Create new engine for each game
                engine = GameManager(seed=game_seed)

                # Clear cache between games for consistent behavior
                agent.clear_cache()

                # Collect game data
                game_data = collect_single_game_data(agent, engine, game_seed, num_simulations)

                # Save game data
                filepath = save_game_data(game_data, output_dir, game_seed)
                saved_files.append(str(filepath))

                score = game_data["metadata"]["final_score"]
                num_turns = game_data["metadata"]["num_turns"]
                scores.append(score)
                total_turns += num_turns

                if verbose:
                    print(f"Score={score}, Turns={num_turns}")

            except Exception as e:
                failed_games.append((game_seed, str(e)))
                if verbose:
                    print(f"FAILED: {e}")
                continue

    except KeyboardInterrupt:
        if verbose:
            print("\nInterrupted by user.")

    # Calculate statistics
    num_collected = len(scores)
    wins = sum(1 for s in scores if s > 0)
    win_percentage = (wins / num_collected) * 100.0 if num_collected > 0 else 0.0
    average_score = sum(scores) / num_collected if num_collected > 0 else 0.0
    best_score = max(scores) if scores else 0
    worst_score = min(scores) if scores else 0
    avg_turns = total_turns / num_collected if num_collected > 0 else 0.0

    generation_time = time.time() - start_time
    games_per_hour = (num_collected / generation_time) * 3600 if generation_time > 0 else 0.0

    results = {
        "num_games": num_collected,
        "wins": wins,
        "win_percentage": win_percentage,
        "average_score": average_score,
        "best_score": best_score,
        "worst_score": worst_score,
        "avg_turns": avg_turns,
        "total_turns": total_turns,
        "games_per_hour": games_per_hour,
        "generation_time": generation_time,
        "output_dir": str(output_dir),
        "saved_files": saved_files,
        "failed_games": failed_games,
        "num_failed": len(failed_games),
        "base_seed": base_seed,
    }

    if verbose:
        print()
        print("=== Self-Play Generation Complete ===")
        print(f"Games Generated: {num_collected}")
        print(f"Wins: {wins} ({win_percentage:.2f}%)")
        print(f"Average Score: {average_score:.2f}")
        print(f"Best Score: {best_score}")
        print(f"Worst Score: {worst_score}")
        print(f"Average Turns: {avg_turns:.1f}")
        print(f"Games/Hour: {games_per_hour:.1f}")
        print(f"Generation Time: {generation_time:.1f}s")
        if failed_games:
            print(f"Failed Games: {len(failed_games)}")
        print(f"Output Directory: {output_dir}")

    return results


def main():
    """Command-line interface for game generation."""
    parser = argparse.ArgumentParser(
        description="Generate self-play games for AlphaGo training."
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=SELF_PLAY_GAMES_PER_ITERATION,
        help=f"Number of games to generate (default: {SELF_PLAY_GAMES_PER_ITERATION})"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=SELF_PLAY_NUM_WORKERS,
        help=f"Number of internal MCTS workers (default: {SELF_PLAY_NUM_WORKERS})"
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=SELF_PLAY_SIMULATIONS,
        help=f"Number of MCTS simulations per move (default: {SELF_PLAY_SIMULATIONS})"
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=None,
        help="Path to policy network checkpoint"
    )
    parser.add_argument(
        "--policy-small-checkpoint",
        type=str,
        default=None,
        help="Path to fast rollout policy checkpoint"
    )
    parser.add_argument(
        "--value-checkpoint",
        type=str,
        default=None,
        help="Path to value network checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save generated games"
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Base seed for game generation (default: random)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information"
    )

    args = parser.parse_args()

    generate_self_play_games(
        num_games=args.num_games,
        num_workers=args.num_workers,
        policy_checkpoint=Path(args.policy_checkpoint) if args.policy_checkpoint else None,
        policy_small_checkpoint=Path(args.policy_small_checkpoint) if args.policy_small_checkpoint else None,
        value_checkpoint=Path(args.value_checkpoint) if args.value_checkpoint else None,
        output_dir=Path(args.output_dir),
        num_simulations=args.num_simulations,
        base_seed=args.base_seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()