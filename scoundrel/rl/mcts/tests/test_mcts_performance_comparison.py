"""
Pytest tests for comparing sequential vs parallel MCTS performance.

These tests compare sequential and parallel execution to measure speedup from parallelization.
"""
import pytest

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.mcts.mcts_agent import MCTSAgent
from scoundrel.rl.mcts.constants import (
    MCTS_NUM_SIMULATIONS,
    MCTS_EXPLORATION_CONSTANT,
    MCTS_MAX_DEPTH,
    USE_RANDOM_ROLLOUT,
    MCTS_NUM_WORKERS,
    MCTS_TEST_NUM_GAMES,
)
from .test_utils import play_games_with_timing


def test_mcts_performance_comparison(game_seed):
    """
    Compare sequential vs parallel MCTS performance.
    
    Runs both sequential and parallel tests and compares the results
    to measure speedup from parallelization.
    """
    # Sequential test
    seq_agent = MCTSAgent(
        num_simulations=MCTS_NUM_SIMULATIONS,
        num_workers=0,
        exploration_constant=MCTS_EXPLORATION_CONSTANT,
        max_depth=MCTS_MAX_DEPTH,
        use_random_rollout=USE_RANDOM_ROLLOUT
    )
    seq_engine = GameManager(seed=game_seed)
    seq_metrics = play_games_with_timing(
        agent=seq_agent,
        engine=seq_engine,
        num_games=MCTS_TEST_NUM_GAMES,
        game_seed=game_seed
    )
    
    # Parallel test
    par_agent = MCTSAgent(
        num_simulations=MCTS_NUM_SIMULATIONS,
        num_workers=MCTS_NUM_WORKERS,
        exploration_constant=MCTS_EXPLORATION_CONSTANT,
        max_depth=MCTS_MAX_DEPTH,
        use_random_rollout=USE_RANDOM_ROLLOUT
    )
    par_engine = GameManager(seed=game_seed)
    par_metrics = play_games_with_timing(
        agent=par_agent,
        engine=par_engine,
        num_games=MCTS_TEST_NUM_GAMES,
        game_seed=game_seed
    )
    
    # Calculate speedup
    speedup = seq_metrics.avg_time_per_step / par_metrics.avg_time_per_step if par_metrics.avg_time_per_step > 0 else 0.0
    
    # Print comparison
    print(f"\n=== Performance Comparison ===")
    print(f"Sequential - Avg time per step: {seq_metrics.avg_time_per_step:.4f}s")
    print(f"Parallel   - Avg time per step: {par_metrics.avg_time_per_step:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Assertions
    assert seq_metrics.total_steps > 0
    assert par_metrics.total_steps > 0
    assert speedup > 0, "Speedup should be positive"
    assert seq_metrics.num_games == MCTS_TEST_NUM_GAMES
    assert par_metrics.num_games == MCTS_TEST_NUM_GAMES
    
    # Note: We don't assert speedup > 1.0 because parallelization overhead
    # might make it slower for small workloads, but we still want to measure it
