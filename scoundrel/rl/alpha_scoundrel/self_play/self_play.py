"""
AlphaGo Self-Play Training Loop.

Implements iterative self-play training with REINFORCE policy gradients:
- Generate self-play games using current best AlphaGoAgent
- Train policy with REINFORCE: Win (+1) reinforces, Loss (-1) discourages
- Train value network on game outcomes
- Evaluate and update best checkpoints
- Repeat with improved networks
"""

import argparse
import multiprocessing as mp
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
from torch.utils.tensorboard import SummaryWriter

from scoundrel.rl.alpha_scoundrel.self_play.constants import (
    SELF_PLAY_GAMES_PER_ITERATION,
    SELF_PLAY_NUM_WORKERS,
    SELF_PLAY_SIMULATIONS,
    SELF_PLAY_PARALLEL_GAMES,
    POLICY_EPOCHS_PER_ITERATION,
    VALUE_EPOCHS_PER_ITERATION,
    BATCH_SIZE,
    EVAL_GAMES,
    EVAL_SEED,
    EVAL_SIMULATIONS,
    CHECKPOINT_BASE_DIR,
    RUNS_BASE_DIR,
    ITERATION_PREFIX,
    POLICY_CHECKPOINT_NAME,
    VALUE_CHECKPOINT_NAME,
    BEST_CHECKPOINT_DIR,
    POLICY_LR,
    POLICY_MAX_GRAD_NORM,
    ENTROPY_COEF,
    REWARD_TYPE,
    VALUE_LR,
    VALUE_MAX_GRAD_NORM,
    TRAIN_VAL_SPLIT,
    STACK_SEQ_LEN,
    POLICY_LARGE_CHECKPOINT,
    POLICY_SMALL_CHECKPOINT,
    VALUE_LARGE_CHECKPOINT,
    TENSORBOARD_LOG_INTERVAL,
    TENSORBOARD_HISTOGRAM_INTERVAL,
    LOG_WEIGHT_HISTOGRAMS,
    LOG_GRADIENT_HISTOGRAMS,
)
from scoundrel.rl.alpha_scoundrel.self_play.game_generator import generate_self_play_games
from scoundrel.rl.alpha_scoundrel.self_play.rl_training import train_policy_rl
from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_agent import AlphaGoAgent
from scoundrel.rl.alpha_scoundrel.alphago_mcts.eval import run_evaluation
from scoundrel.rl.alpha_scoundrel.value.value_large.train import train_epoch as train_value_epoch, validate as validate_value
from scoundrel.rl.alpha_scoundrel.value.value_large.network import ValueLargeNet
from scoundrel.rl.alpha_scoundrel.value.value_large.data_loader import create_dataloaders as create_value_dataloaders
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.utils import get_device
from scoundrel.game.game_manager import GameManager


def _save_iteration_summary(
    iteration_dir: Path,
    generation_stats: Dict[str, Any],
    policy_metrics: Dict[str, float],
    value_metrics: Dict[str, float],
    eval_results: Dict[str, Any],
    iteration: int,
    timestamp: str,
    improved: bool = False,
    best_eval_score: float = 0.0,
):
    """Save iteration summary files."""
    results_path = iteration_dir / "results.txt"
    with open(results_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Self-Play Iteration Results Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Training Mode: REINFORCE\n")
        f.write(f"New Best: {'YES ✓' if improved else 'NO'}\n")
        f.write(f"Best Eval Score: {best_eval_score:.2f}\n\n")

        # Game generation stats
        f.write("-" * 60 + "\n")
        f.write("Game Generation\n")
        f.write("-" * 60 + "\n")
        f.write(f"Games Generated: {generation_stats['num_games']}\n")
        f.write(f"Wins: {generation_stats['wins']} ({generation_stats['win_percentage']:.2f}%)\n")
        f.write(f"Average Score: {generation_stats['average_score']:.2f}\n")
        f.write(f"Best Score: {generation_stats['best_score']}\n")
        f.write(f"Worst Score: {generation_stats['worst_score']}\n")
        f.write(f"Average Turns: {generation_stats['avg_turns']:.1f}\n")
        f.write(f"Games/Hour: {generation_stats['games_per_hour']:.1f}\n")
        f.write(f"Generation Time: {generation_stats['generation_time']:.1f}s\n\n")

        # Policy training (REINFORCE)
        f.write("-" * 60 + "\n")
        f.write("Policy Training (REINFORCE)\n")
        f.write("-" * 60 + "\n")
        if policy_metrics:
            f.write(f"Best Epoch: {policy_metrics.get('best_epoch', 'N/A')}\n")
            f.write(f"Best Loss: {policy_metrics.get('best_loss', 'N/A'):.6f}\n")
            f.write(f"Best Accuracy: {policy_metrics.get('best_accuracy', 'N/A'):.4f}\n")
            f.write(f"Final Loss: {policy_metrics.get('loss', 'N/A'):.6f}\n")
            f.write(f"Final Accuracy: {policy_metrics.get('accuracy', 'N/A'):.4f}\n")
            f.write(f"Win Accuracy: {policy_metrics.get('win_accuracy', 'N/A'):.4f}\n")
            f.write(f"Loss Accuracy: {policy_metrics.get('loss_accuracy', 'N/A'):.4f}\n")
            f.write(f"Entropy: {policy_metrics.get('entropy', 'N/A'):.4f}\n")
            f.write(f"Training Time: {policy_metrics.get('training_time', 'N/A'):.1f}s\n")
        f.write("\n")

        # Value training
        f.write("-" * 60 + "\n")
        f.write("Value Training\n")
        f.write("-" * 60 + "\n")
        if value_metrics:
            f.write(f"Best Epoch: {value_metrics.get('best_epoch', 'N/A')}\n")
            f.write(f"Best MSE: {value_metrics.get('best_mse', 'N/A'):.6f}\n")
            f.write(f"Final Loss: {value_metrics.get('loss', 'N/A'):.6f}\n")
            f.write(f"Final MSE: {value_metrics.get('mse', 'N/A'):.6f}\n")
            f.write(f"Final MAE: {value_metrics.get('mae', 'N/A'):.6f}\n")
            f.write(f"Final RMSE: {value_metrics.get('rmse', 'N/A'):.6f}\n")
            f.write(f"Training Time: {value_metrics.get('training_time', 'N/A'):.1f}s\n")
        f.write("\n")

        # Evaluation
        f.write("-" * 60 + "\n")
        f.write("Evaluation\n")
        f.write("-" * 60 + "\n")
        f.write(f"Games Played: {eval_results['num_games']}\n")
        f.write(f"Wins: {eval_results['wins']} ({eval_results['win_percentage']:.2f}%)\n")
        f.write(f"Average Score: {eval_results['average_score']:.2f}\n")
        f.write(f"Best Score: {eval_results['best_score']}\n")
        f.write(f"Worst Score: {eval_results['worst_score']}\n")
        
        f.write("\n" + "=" * 60 + "\n")


def _setup_iteration_directory(base_dir: Path, iteration: int) -> Tuple[Path, Path]:
    """Set up directories for this iteration."""
    iteration_dir = base_dir / f"{ITERATION_PREFIX}{iteration:03d}"
    games_dir = iteration_dir / "games"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    games_dir.mkdir(parents=True, exist_ok=True)
    return iteration_dir, games_dir


def _copy_best_checkpoints(src_dir: Path, dst_dir: Path):
    """Copy best checkpoints to a new location."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    policy_src = src_dir / POLICY_CHECKPOINT_NAME
    policy_dst = dst_dir / POLICY_CHECKPOINT_NAME
    if policy_src.exists():
        shutil.copy2(policy_src, policy_dst)

    value_src = src_dir / VALUE_CHECKPOINT_NAME
    value_dst = dst_dir / VALUE_CHECKPOINT_NAME
    if value_src.exists():
        shutil.copy2(value_src, value_dst)


def _load_best_checkpoints(base_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Load paths to best checkpoints."""
    best_dir = base_dir / BEST_CHECKPOINT_DIR
    policy_checkpoint = best_dir / POLICY_CHECKPOINT_NAME
    value_checkpoint = best_dir / VALUE_CHECKPOINT_NAME

    policy_path = policy_checkpoint if policy_checkpoint.exists() else None
    value_path = value_checkpoint if value_checkpoint.exists() else None

    return policy_path, value_path


def _initialize_best_checkpoints(
    base_dir: Path,
    initial_policy_checkpoint: Optional[Path] = None,
    initial_value_checkpoint: Optional[Path] = None,
):
    """Initialize best checkpoints from initial checkpoints."""
    best_dir = base_dir / BEST_CHECKPOINT_DIR
    best_dir.mkdir(parents=True, exist_ok=True)

    policy_dst = best_dir / POLICY_CHECKPOINT_NAME
    value_dst = best_dir / VALUE_CHECKPOINT_NAME

    # Use provided checkpoints or fall back to defaults
    policy_src = initial_policy_checkpoint if initial_policy_checkpoint else POLICY_LARGE_CHECKPOINT
    value_src = initial_value_checkpoint if initial_value_checkpoint else VALUE_LARGE_CHECKPOINT

    if policy_src and policy_src.exists():
        shutil.copy2(policy_src, policy_dst)
    if value_src and value_src.exists():
        shutil.copy2(value_src, value_dst)


def _train_value_network(
    games_dir: Path,
    checkpoint_path: Path,
    pretrained_checkpoint: Optional[Path],
    writer: Optional[SummaryWriter],
    iteration: int,
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, float]:
    """Train value network on self-play game outcomes."""
    training_start_time = time.time()
    
    translator = ScoundrelTranslator(stack_seq_len=STACK_SEQ_LEN)
    dummy_state, _ = translator.encode_state(GameManager().restart())
    scalar_input_dim = dummy_state.shape[1]

    train_loader, val_loader = create_value_dataloaders(
        log_dir=games_dir,
        translator=translator,
        batch_size=BATCH_SIZE,
        train_val_split=TRAIN_VAL_SPLIT,
        max_games=None,
    )

    model = ValueLargeNet(scalar_input_dim=scalar_input_dim).to(device)
    
    # Log model info
    if writer is not None:
        total_params = sum(p.numel() for p in model.parameters())
        writer.add_scalar('Value/TotalParams', total_params, 0)
        writer.add_scalar('Value/TrainSamples', len(train_loader.dataset), 0)
        writer.add_scalar('Value/ValSamples', len(val_loader.dataset), 0)

    pretrained_mse = None
    if pretrained_checkpoint and pretrained_checkpoint.exists():
        if verbose:
            print(f"  Loading pretrained value weights from: {pretrained_checkpoint}")
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_mse = checkpoint.get('val_mse', None)
        if verbose:
            print(f"    Loaded checkpoint with val_mse: {pretrained_mse}")
        
        if writer is not None:
            writer.add_text(f'Value/PretrainedFrom_Iter{iteration}', str(pretrained_checkpoint), 0)
    else:
        if verbose:
            print("  No pretrained value checkpoint found, training from scratch")

    optimizer = torch.optim.Adam(model.parameters(), lr=VALUE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VALUE_EPOCHS_PER_ITERATION, eta_min=VALUE_LR * 0.1)

    best_val_mse = float('inf')
    best_epoch = 0
    final_metrics = {}

    if verbose:
        print(f"  Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    for epoch in range(VALUE_EPOCHS_PER_ITERATION):
        global_epoch = epoch + iteration * VALUE_EPOCHS_PER_ITERATION
        
        train_metrics = train_value_epoch(model, train_loader, optimizer, device, writer, epoch)
        val_metrics = validate_value(model, val_loader, device, writer, epoch)
        final_metrics = val_metrics
        
        scheduler.step()

        if verbose:
            print(f"  Epoch {epoch+1}/{VALUE_EPOCHS_PER_ITERATION}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val MSE: {val_metrics['mse']:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'global_epoch': global_epoch,
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_mse': val_metrics['mse'],
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics.get('rmse', 0.0),
                'train_loss': train_metrics['loss'],
                'train_mse': train_metrics.get('mse', 0.0),
                'scalar_input_dim': scalar_input_dim,
                'pretrained_checkpoint': str(pretrained_checkpoint) if pretrained_checkpoint else None,
                'pretrained_mse': pretrained_mse,
            }, checkpoint_path)
            
            if writer is not None:
                writer.add_scalar('Value/BestValMSE', best_val_mse, epoch)

    training_time = time.time() - training_start_time
    
    if verbose:
        print(f"  Best value - Epoch: {best_epoch+1}, MSE: {best_val_mse:.4f}")
        print(f"  Value training completed in {training_time:.1f}s")
    
    # Include additional info in returned metrics
    final_metrics['best_mse'] = best_val_mse
    final_metrics['best_epoch'] = best_epoch
    final_metrics['training_time'] = training_time

    return final_metrics


def _evaluate_single_game_worker_with_progress(
    worker_id: int,
    game_num: int,
    base_seed: int,
    policy_checkpoint: str,
    value_checkpoint: str,
    num_simulations: int,
    num_workers: int,
    progress_queue,
    result_queue,
) -> None:
    """
    Worker function for parallel evaluation of a single game with progress updates.

    Args:
        worker_id: Unique identifier for this worker
        game_num: Game number (used for seed offset)
        base_seed: Base seed for reproducibility
        policy_checkpoint: Path to policy checkpoint
        value_checkpoint: Path to value checkpoint
        num_simulations: MCTS simulations per move
        num_workers: MCTS internal workers
        progress_queue: Queue for sending progress updates
        result_queue: Queue for sending final results
    """
    try:
        from scoundrel.game.game_manager import GameManager
        from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_agent import AlphaGoAgent

        engine_seed = base_seed + game_num
        engine = GameManager(seed=engine_seed)

        agent = AlphaGoAgent(
            policy_large_checkpoint=policy_checkpoint,
            value_checkpoint=value_checkpoint,
            num_simulations=num_simulations,
            num_workers=num_workers,
        )

        # Send initial progress update
        progress_queue.put(('start', worker_id, engine_seed, 0))

        state = engine.restart()
        turn = 0

        while not state.game_over:
            # Send progress update for current turn
            progress_queue.put(('turn', worker_id, engine_seed, turn))

            action_idx = agent.select_action(state)
            action_enum = agent.translator.decode_action(action_idx)

            engine.execute_turn(action_enum)
            state = engine.get_state()

            turn += 1

        final_score = state.score

        # Send completion update
        progress_queue.put(('complete', worker_id, engine_seed, turn))

        # Send result
        result_queue.put({
            'worker_id': worker_id,
            'game_num': game_num,
            'game_seed': engine_seed,
            'score': final_score,
            'success': True,
        })

    except Exception as e:
        # Send error result
        result_queue.put({
            'worker_id': worker_id,
            'game_num': game_num,
            'game_seed': base_seed + game_num,
            'error': str(e),
            'success': False,
        })


def _evaluate_single_game_worker(
    game_num: int,
    base_seed: int,
    policy_checkpoint: str,
    value_checkpoint: str,
    num_simulations: int,
    num_workers: int,
) -> int:
    """
    Worker function for parallel evaluation of a single game.

    Args:
        game_num: Game number (used for seed offset)
        base_seed: Base seed for reproducibility
        policy_checkpoint: Path to policy checkpoint
        value_checkpoint: Path to value checkpoint
        num_simulations: MCTS simulations per move
        num_workers: MCTS internal workers

    Returns:
        Game score
    """
    from scoundrel.game.game_manager import GameManager
    from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_agent import AlphaGoAgent

    engine_seed = base_seed + game_num
    engine = GameManager(seed=engine_seed)

    agent = AlphaGoAgent(
        policy_large_checkpoint=policy_checkpoint,
        value_checkpoint=value_checkpoint,
        num_simulations=num_simulations,
        num_workers=num_workers,
    )

    state = engine.restart()
    while not state.game_over:
        action_idx = agent.select_action(state)
        action_enum = agent.translator.decode_action(action_idx)
        engine.execute_turn(action_enum)
        state = engine.get_state()

    return state.score


def _evaluate_agent_parallel(
    policy_checkpoint: Optional[Path],
    value_checkpoint: Optional[Path],
    num_games: int = EVAL_GAMES,
    seed: int = EVAL_SEED,
    num_parallel_games: int = SELF_PLAY_PARALLEL_GAMES,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate an AlphaGo agent using parallel game execution.

    Args:
        policy_checkpoint: Path to policy network checkpoint
        value_checkpoint: Path to value network checkpoint
        num_games: Total number of games to evaluate (EVAL_GAMES)
        seed: Base seed for reproducibility
        num_parallel_games: Number of games to run simultaneously (SELF_PLAY_PARALLEL_GAMES)
        verbose: Whether to print live progress with AlphaGo MCTS collection format

    Returns:
        Dictionary with evaluation results
    """
    try:
        # Convert Path objects to strings for multiprocessing
        policy_path = str(policy_checkpoint) if policy_checkpoint else None
        value_path = str(value_checkpoint) if value_checkpoint else None

        scores = []
        wins = 0

        # Use multiprocessing with queues for progress tracking (like game generation)
        progress_queue = mp.Queue()
        result_queue = mp.Queue()

        if verbose:
            print("=== AlphaGo Self-Play Evaluation ===")
            print(f"Running {num_games} evaluation games using {num_parallel_games} parallel workers...")
            print(f"AlphaGo MCTS configuration:")
            print(f"  Simulations per move: {EVAL_SIMULATIONS}")
            print(f"  Internal MCTS workers: {SELF_PLAY_NUM_WORKERS}")
            print(f"  Policy checkpoint: {policy_checkpoint}")
            print(f"  Value checkpoint: {value_checkpoint}")
            print()

        # Track evaluation progress
        actual_completed_games = 0
        next_game_num = 0
        scores = [None] * num_games
        active_games = {}  # worker_id -> (game_seed, start_time, current_turn)
        completed_game_history = []  # List of (game_seed, score, game_num) tuples

        # Create worker processes
        available_workers = list(range(num_parallel_games))
        active_processes = {}  # worker_id -> process

        def start_new_game(worker_id: int) -> bool:
            """Start a new evaluation game for the given worker. Returns False if no more games to start."""
            nonlocal next_game_num

            if next_game_num >= num_games:
                return False

            game_num = next_game_num
            next_game_num += 1
            game_seed = seed + game_num

            # Start process for this evaluation game
            process = mp.Process(
                target=_evaluate_single_game_worker_with_progress,
                args=(worker_id, game_num, seed, policy_path, value_path, EVAL_SIMULATIONS, SELF_PLAY_NUM_WORKERS, progress_queue, result_queue)
            )
            process.start()
            active_processes[worker_id] = process

            return True

        def print_verbose_status():
            """Print the current status with history and active games - exact format from AlphaGo MCTS collection."""
            # Clear screen and move cursor to top
            print("\033[2J\033[H", end="")  # Clear screen and move to top

            # Print header
            print("=== AlphaGo Self-Play Evaluation ===")
            print(f"Output: evaluation")
            print(f"Parallel games: {num_parallel_games} | Completed: {len(completed_game_history)}")
            if num_games is not None:
                print(f"Target: {num_games} games")
            print()

            # Print completed games history (last 10)
            if completed_game_history:
                print("=== Completed Games ===")
                for i, (game_seed, score, worker_id) in enumerate(completed_game_history[-10:]):  # Show last 10
                    status = "WIN" if score > 0 else "LOSE"
                    color = "\033[92m" if score > 0 else "\033[91m"  # Green for win, red for loss
                    print(f"{color}Game {game_seed}: {status} Score={score}, Worker={worker_id}\033[0m")
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

        try:
            # Start initial batch of games
            for worker_id in available_workers[:]:
                if not start_new_game(worker_id):
                    available_workers.remove(worker_id)

            # Main evaluation loop
            while active_processes or actual_completed_games < num_games:
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
                    except:
                        pass

                # Check for completed results
                while not result_queue.empty():
                    try:
                        result = result_queue.get_nowait()

                        actual_completed_games += 1

                        if result['success']:
                            game_num = result['game_num']
                            scores[game_num] = result['score']
                            completed_game_history.append((
                                result['game_seed'],
                                result['score'],
                                result['worker_id']
                            ))
                        else:
                            # Handle failed games
                            game_num = result.get('game_num', 0)
                            scores[game_num] = 0  # Default to loss
                            completed_game_history.append((
                                result.get('game_seed', seed + game_num),
                                0,
                                result['worker_id']
                            ))

                        # Clean up finished process
                        if result['worker_id'] in active_processes:
                            active_processes[result['worker_id']].join()
                            del active_processes[result['worker_id']]

                        # Start new game for this worker if more games needed
                        if actual_completed_games < num_games:
                            start_new_game(result['worker_id'])

                    except:
                        pass

                if verbose:
                    print_verbose_status()

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

        if verbose:
            print_verbose_status()  # Final status update
            print("=== Evaluation Complete ===")
            wins = sum(1 for score in scores if score > 0)
            win_percentage = (wins / num_games) * 100.0 if num_games > 0 else 0.0
            average_score = sum(scores) / num_games if num_games > 0 else 0.0
            print(f"Games: {num_games}")
            print(f"Wins: {wins} ({win_percentage:.2f}%)")
            print(f"Average Score: {average_score:.2f}")
            print(f"Best Score: {max(scores) if scores else 0}")
            print(f"Worst Score: {min(scores) if scores else 0}")
            print()

        # Calculate statistics
        wins = sum(1 for score in scores if score > 0)
        win_percentage = (wins / num_games) * 100.0 if num_games > 0 else 0.0

        total_score = sum(scores)
        average_score = total_score / num_games if num_games > 0 else 0.0

        best_score = max(scores) if scores else 0
        worst_score = min(scores) if scores else 0

        return {
            "num_games": num_games,
            "wins": wins,
            "win_percentage": win_percentage,
            "average_score": average_score,
            "best_score": best_score,
            "worst_score": worst_score,
            "total_score": total_score,
            "scores": scores,
        }

    except Exception as e:
        if verbose:
            print(f"Parallel evaluation failed: {e}")
        return {
            "num_games": 0,
            "wins": 0,
            "win_percentage": 0,
            "average_score": 0,
            "best_score": 0,
            "worst_score": 0,
            "total_score": 0,
            "scores": [],
        }


def print_evaluation_results(results: dict, show_individual_games: bool = True):
    """Print self-play evaluation results in formatted output."""
    print(f"\n=== Self-Play AlphaGo MCTS Evaluation Results ===")
    print(f"Configuration:")
    print(f"  Simulations per move: {EVAL_SIMULATIONS}")
    print(f"  Internal MCTS workers: {SELF_PLAY_NUM_WORKERS}")
    print(f"  Parallel games: {SELF_PLAY_PARALLEL_GAMES}")
    print(f"\nResults:")
    print(f"  Games Played: {results['num_games']}")
    print(f"  Wins: {results['wins']}")
    print(f"  Win Percentage: {results['win_percentage']:.2f}%")
    print(f"  Average Score: {results['average_score']:.2f}")
    print(f"  Best Score: {results['best_score']}")
    print(f"  Worst Score: {results['worst_score']}")
    print(f"  Total Score: {results['total_score']}")

    # Show individual game results with colors
    if show_individual_games and 'scores' in results:
        print(f"\nIndividual Games:")
        for game_num, score in enumerate(results['scores']):
            status = "WIN" if score > 0 else "LOSE"
            color = "\033[92m" if score > 0 else "\033[91m"  # Green for win, red for loss
            print(f"{color}Game {game_num + 1}: {status} Score={score}\033[0m")


def _evaluate_agent(
    policy_checkpoint: Optional[Path],
    value_checkpoint: Optional[Path],
    num_games: int = EVAL_GAMES,
    seed: int = EVAL_SEED,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate an AlphaGo agent using parallel game execution with live progress display."""
    return _evaluate_agent_parallel(
        policy_checkpoint=policy_checkpoint,
        value_checkpoint=value_checkpoint,
        num_games=num_games,
        seed=seed,
        num_parallel_games=SELF_PLAY_PARALLEL_GAMES,
        verbose=verbose,
    )


def run_self_play_training(
    max_iterations: int = 10,
    checkpoint_base_dir: Path = CHECKPOINT_BASE_DIR,
    tensorboard: bool = True,
    resume_from: Optional[int] = None,
    initial_policy_checkpoint: Optional[Path] = None,
    initial_value_checkpoint: Optional[Path] = None,
    num_parallel_games: int = SELF_PLAY_PARALLEL_GAMES,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the self-play training loop with REINFORCE policy gradients.

    Args:
        max_iterations: Maximum number of iterations to run
        checkpoint_base_dir: Base directory for checkpoints
        tensorboard: Enable TensorBoard logging
        resume_from: Resume from specific iteration (None = start from 1)
        initial_policy_checkpoint: Path to initial policy checkpoint (overrides default)
        initial_value_checkpoint: Path to initial value checkpoint (overrides default)
        num_parallel_games: Number of games to generate simultaneously
        verbose: Print detailed progress

    Returns:
        Training summary
    """
    device = get_device()
    start_time = time.time()
    
    # Resolve checkpoint paths
    policy_init = initial_policy_checkpoint if initial_policy_checkpoint else POLICY_LARGE_CHECKPOINT
    value_init = initial_value_checkpoint if initial_value_checkpoint else VALUE_LARGE_CHECKPOINT

    if verbose:
        print(f"{'='*60}")
        print(f"AlphaGo Self-Play Training with REINFORCE")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Checkpoint base directory: {checkpoint_base_dir}")
        print(f"Training mode: REINFORCE (win=+1, loss=-1)")
        print(f"Max iterations: {max_iterations}")
        print(f"\nInitial checkpoints:")
        print(f"  Policy: {policy_init}")
        print(f"  Value:  {value_init}")
        print(f"{'='*60}")

    checkpoint_base_dir = Path(checkpoint_base_dir)
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)

    # Set up TensorBoard
    writer = None
    log_dir = None
    if tensorboard:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"self_play_{run_timestamp}"
        log_dir = RUNS_BASE_DIR / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        if verbose:
            print(f"TensorBoard logs: {log_dir}")
        
        # Log hyperparameters at start
        writer.add_hparams(
            {
                'max_iterations': max_iterations,
                'games_per_iteration': SELF_PLAY_GAMES_PER_ITERATION,
                'self_play_simulations': SELF_PLAY_SIMULATIONS,
                'policy_epochs': POLICY_EPOCHS_PER_ITERATION,
                'value_epochs': VALUE_EPOCHS_PER_ITERATION,
                'batch_size': BATCH_SIZE,
                'policy_lr': POLICY_LR,
                'value_lr': VALUE_LR,
                'entropy_coef': ENTROPY_COEF,
                'reward_type': REWARD_TYPE,
                'eval_games': EVAL_GAMES,
                'eval_simulations': EVAL_SIMULATIONS,
            },
            {},
            run_name=run_name,
        )
        
        # Log initial checkpoint paths
        writer.add_text('Config/InitialPolicyCheckpoint', str(policy_init), 0)
        writer.add_text('Config/InitialValueCheckpoint', str(value_init), 0)
        writer.add_text('Config/CheckpointDir', str(checkpoint_base_dir), 0)

    # Initialize best checkpoints if not resuming
    start_iteration = resume_from if resume_from is not None else 1
    if start_iteration == 1:
        _initialize_best_checkpoints(
            checkpoint_base_dir,
            initial_policy_checkpoint=policy_init,
            initial_value_checkpoint=value_init,
        )
        if verbose:
            print("Initialized best checkpoints from initial models")

    # Evaluate the current best agent to establish a baseline score
    policy_best, value_best = _load_best_checkpoints(checkpoint_base_dir)
    best_eval_score = float('-inf')

    if policy_best and value_best:
        if verbose:
            print(f"\nEvaluating current best agent to establish baseline score...")
            print(f"Policy: {policy_best}")
            print(f"Value:  {value_best}")
        
        baseline_results = _evaluate_agent(
            policy_checkpoint=policy_best,
            value_checkpoint=value_best,
            num_games=EVAL_GAMES,
            seed=EVAL_SEED,
            verbose=verbose,
        )

        if verbose:
            print_evaluation_results(baseline_results, show_individual_games=True)

        best_eval_score = baseline_results['average_score']
        if verbose:
            print(f"Baseline score established: {best_eval_score:.2f}")
            if writer:
                writer.add_scalar('Progress/BestEvalScore', best_eval_score, start_iteration - 1)

    iteration = start_iteration

    try:
        for iteration in range(start_iteration, max_iterations + 1):
            iteration_start_time = time.time()
            if verbose:
                print(f"\n{'='*50}")
                print(f"Starting Iteration {iteration}")
                print(f"{'='*50}")

            iteration_dir, games_dir = _setup_iteration_directory(checkpoint_base_dir, iteration)
            policy_checkpoint, value_checkpoint = _load_best_checkpoints(checkpoint_base_dir)

            # Create iteration-specific TensorBoard writer
            iter_writer = None
            if tensorboard and log_dir:
                iter_log_dir = log_dir / f"iter_{iteration:03d}"
                iter_writer = SummaryWriter(log_dir=str(iter_log_dir))

            if verbose:
                print(f"Using policy checkpoint: {policy_checkpoint}")
                print(f"Using value checkpoint: {value_checkpoint}")

            # 1. Generate self-play games
            if verbose:
                print("\nStep 1: Generating self-play games...")
            generation_stats = generate_self_play_games(
                num_games=SELF_PLAY_GAMES_PER_ITERATION,
                num_workers=SELF_PLAY_NUM_WORKERS,
                policy_checkpoint=policy_checkpoint,
                policy_small_checkpoint=POLICY_SMALL_CHECKPOINT,
                value_checkpoint=value_checkpoint,
                output_dir=games_dir,
                num_simulations=SELF_PLAY_SIMULATIONS,
                num_parallel_games=num_parallel_games,
                verbose=verbose,
            )

            if writer:
                step = iteration
                writer.add_scalar('SelfPlay/GamesGenerated', generation_stats['num_games'], step)
                writer.add_scalar('SelfPlay/WinPercentage', generation_stats['win_percentage'], step)
                writer.add_scalar('SelfPlay/AverageScore', generation_stats['average_score'], step)
                writer.add_scalar('SelfPlay/BestScore', generation_stats['best_score'], step)
                writer.add_scalar('SelfPlay/AverageTurns', generation_stats['avg_turns'], step)
                writer.add_scalar('SelfPlay/GamesPerHour', generation_stats['games_per_hour'], step)
                writer.add_scalar('SelfPlay/GenerationTime', generation_stats['generation_time'], step)

            # 2. Train policy network with REINFORCE
            if verbose:
                print("\nStep 2: Training policy network (REINFORCE)...")
            policy_checkpoint_path = iteration_dir / POLICY_CHECKPOINT_NAME
            policy_metrics = train_policy_rl(
                games_dir=games_dir,
                checkpoint_path=policy_checkpoint_path,
                pretrained_checkpoint=policy_checkpoint,
                writer=iter_writer,
                iteration=iteration,
                device=device,
                epochs=POLICY_EPOCHS_PER_ITERATION,
                batch_size=BATCH_SIZE,
                lr=POLICY_LR,
                max_grad_norm=POLICY_MAX_GRAD_NORM,
                entropy_coef=ENTROPY_COEF,
                reward_type=REWARD_TYPE,
                stack_seq_len=STACK_SEQ_LEN,
                verbose=verbose,
            )
            
            if writer:
                writer.add_scalar('Policy/TrainingTime', policy_metrics.get('training_time', 0), iteration)
                writer.add_scalar('Policy/FinalBestLoss', policy_metrics.get('best_loss', 0), iteration)
                writer.add_scalar('Policy/FinalBestAccuracy', policy_metrics.get('best_accuracy', 0), iteration)
                writer.add_scalar('Policy/BestEpoch', policy_metrics.get('best_epoch', 0), iteration)

            # 3. Train value network
            if verbose:
                print("\nStep 3: Training value network...")
            value_checkpoint_path = iteration_dir / VALUE_CHECKPOINT_NAME
            value_metrics = _train_value_network(
                games_dir=games_dir,
                checkpoint_path=value_checkpoint_path,
                pretrained_checkpoint=value_checkpoint,
                writer=iter_writer,
                iteration=iteration,
                device=device,
                verbose=verbose,
            )
            
            if writer:
                writer.add_scalar('Value/TrainingTime', value_metrics.get('training_time', 0), iteration)
                writer.add_scalar('Value/FinalBestMSE', value_metrics.get('best_mse', 0), iteration)
                writer.add_scalar('Value/BestEpoch', value_metrics.get('best_epoch', 0), iteration)
            
            if iter_writer:
                iter_writer.close()

            # 4. Evaluate new agent
            if verbose:
                print("\nStep 4: Evaluating new agent...")
            eval_results = _evaluate_agent(
                policy_checkpoint=policy_checkpoint_path,
                value_checkpoint=value_checkpoint_path,
                num_games=EVAL_GAMES,
                seed=EVAL_SEED,
                verbose=verbose,
            )

            if verbose:
                print_evaluation_results(eval_results, show_individual_games=False)

            if writer:
                writer.add_scalar('Evaluation/WinPercentage', eval_results['win_percentage'], iteration)
                writer.add_scalar('Evaluation/AverageScore', eval_results['average_score'], iteration)
                writer.add_scalar('Evaluation/BestScore', eval_results['best_score'], iteration)
                writer.add_scalar('Evaluation/WorstScore', eval_results['worst_score'], iteration)
                writer.add_scalar('Evaluation/TotalScore', eval_results.get('total_score', 0), iteration)
                writer.add_scalar('Evaluation/NumGames', eval_results['num_games'], iteration)

            # 5. Update best checkpoints if improved
            current_eval_score = eval_results['average_score']
            improved = current_eval_score > best_eval_score

            if improved:
                best_eval_score = current_eval_score
                _copy_best_checkpoints(iteration_dir, checkpoint_base_dir / BEST_CHECKPOINT_DIR)
                if verbose:
                    print(f"✓ New best agent! Average score: {current_eval_score:.2f}")
            else:
                if verbose:
                    print(f"✗ Agent did not improve. Best score remains: {best_eval_score:.2f}")
            
            # Log improvement metrics
            if writer:
                writer.add_scalar('Progress/Improved', float(improved), iteration)
                writer.add_scalar('Progress/BestEvalScore', best_eval_score, iteration)
                writer.add_scalar('Progress/CurrentEvalScore', current_eval_score, iteration)
                writer.add_scalar('Progress/ImprovementGap', current_eval_score - best_eval_score if not improved else 0, iteration)

            # Save iteration summary
            timestamp = datetime.now().isoformat()
            _save_iteration_summary(
                iteration_dir, generation_stats, policy_metrics,
                value_metrics, eval_results, iteration, timestamp,
                improved=improved, best_eval_score=best_eval_score
            )

            iteration_time = time.time() - iteration_start_time
            
            # Log iteration timing
            if writer:
                writer.add_scalar('Timing/IterationTime', iteration_time, iteration)
                writer.add_scalar('Timing/GenerationTime', generation_stats['generation_time'], iteration)
                writer.add_scalar('Timing/PolicyTrainingTime', policy_metrics.get('training_time', 0), iteration)
                writer.add_scalar('Timing/ValueTrainingTime', value_metrics.get('training_time', 0), iteration)
                
                # Log combined metrics
                total_samples = len(list(games_dir.glob("*.json")))
                writer.add_scalar('Progress/TotalGamesGenerated', total_samples, iteration)
                writer.add_scalar('Progress/IterationsCompleted', iteration, iteration)
            
            if verbose:
                print(f"\nIteration {iteration} completed in {iteration_time:.1f}s")
                print(f"Evaluation score: {current_eval_score:.2f} (best: {best_eval_score:.2f})")

    except KeyboardInterrupt:
        if verbose:
            print("\n\nTraining interrupted by user (Ctrl+C)")
            print("Saving final state...")

    finally:
        if writer:
            writer.flush()
            writer.close()

    total_time = time.time() - start_time
    if verbose:
        print(f"\n{'='*50}")
        print("Self-play training completed!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Iterations completed: {iteration}")
        print(f"Best evaluation score: {best_eval_score:.2f}")
        print(f"Checkpoints saved to: {checkpoint_base_dir}")
        print(f"{'='*50}")

    return {
        "iterations_completed": iteration,
        "best_eval_score": best_eval_score,
        "total_time": total_time,
        "checkpoint_dir": str(checkpoint_base_dir),
        "tensorboard_dir": str(log_dir) if tensorboard else None,
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run AlphaGo self-play training with REINFORCE policy gradients.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start training with default checkpoints
  python -m scoundrel.rl.alpha_scoundrel.self_play.self_play -v

  # Start with custom checkpoints
  python -m scoundrel.rl.alpha_scoundrel.self_play.self_play -v \\
    --policy-checkpoint path/to/policy.pt \\
    --value-checkpoint path/to/value.pt

  # Resume from iteration 5
  python -m scoundrel.rl.alpha_scoundrel.self_play.self_play -v --resume-from 5

  # Run for 50 iterations
  python -m scoundrel.rl.alpha_scoundrel.self_play.self_play -v --max-iterations 50
"""
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum number of iterations to run (default: 10)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(CHECKPOINT_BASE_DIR),
        help=f"Base directory for checkpoints (default: {CHECKPOINT_BASE_DIR})"
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=None,
        help="Resume training from specific iteration (default: start from 1)"
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=None,
        help=f"Initial policy checkpoint to start from (default: {POLICY_LARGE_CHECKPOINT})"
    )
    parser.add_argument(
        "--value-checkpoint",
        type=str,
        default=None,
        help=f"Initial value checkpoint to start from (default: {VALUE_LARGE_CHECKPOINT})"
    )
    parser.add_argument(
        "--policy-small-checkpoint",
        type=str,
        default=None,
        help=f"Policy small checkpoint for fast rollouts (default: {POLICY_SMALL_CHECKPOINT})"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    parser.add_argument(
        "--num-parallel-games",
        type=int,
        default=SELF_PLAY_PARALLEL_GAMES,
        help=f"Number of games to generate simultaneously (default: {SELF_PLAY_PARALLEL_GAMES})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information with colored concurrent game status"
    )

    args = parser.parse_args()
    
    # Parse checkpoint paths
    policy_checkpoint = Path(args.policy_checkpoint) if args.policy_checkpoint else None
    value_checkpoint = Path(args.value_checkpoint) if args.value_checkpoint else None

    run_self_play_training(
        max_iterations=args.max_iterations,
        checkpoint_base_dir=Path(args.checkpoint_dir),
        tensorboard=not args.no_tensorboard,
        resume_from=args.resume_from,
        initial_policy_checkpoint=policy_checkpoint,
        initial_value_checkpoint=value_checkpoint,
        num_parallel_games=args.num_parallel_games,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
