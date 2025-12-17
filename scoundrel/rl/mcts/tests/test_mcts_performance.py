"""
Pytest test for MCTS performance benchmarking with multithreading.

This test measures intrinsic time per game step using multithreading to evaluate
the best intrinsic efficiency of end-to-end gameplay.
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


def test_mcts_multithreaded_performance(game_seed):
    """
    Test multithreaded MCTS performance for best intrinsic efficiency.
    
    Measures time per step with parallel execution using multithreading.
    This test evaluates the efficiency gains from multithreading in end-to-end gameplay.
    """
    # Create agent with parallel execution
    agent = MCTSAgent(
        num_simulations=MCTS_NUM_SIMULATIONS,
        num_workers=MCTS_NUM_WORKERS,
        exploration_constant=MCTS_EXPLORATION_CONSTANT,
        max_depth=MCTS_MAX_DEPTH,
        use_random_rollout=USE_RANDOM_ROLLOUT
    )
    
    # Create game manager (will be recreated per game with deterministic seeds)
    engine = GameManager(seed=game_seed)
    
    # Run test and collect metrics
    metrics = play_games_with_timing(
        agent=agent,
        engine=engine,
        num_games=MCTS_TEST_NUM_GAMES,
        game_seed=game_seed
    )
    
    # Assertions
    assert metrics.total_steps > 0, "Should have played at least one step"
    assert metrics.avg_time_per_step > 0, "Average time per step should be positive"
    assert metrics.num_workers == MCTS_NUM_WORKERS, "Should be parallel execution"
    assert metrics.num_simulations == MCTS_NUM_SIMULATIONS, "Should use configured simulations"
    assert metrics.num_games == MCTS_TEST_NUM_GAMES, f"Should have played {MCTS_TEST_NUM_GAMES} games"
    
    # Print metrics for visibility
    print(f"\n=== Multithreaded MCTS Performance ({MCTS_NUM_WORKERS} workers) ===")
    print(f"Games played: {metrics.num_games}")
    print(f"Total steps: {metrics.total_steps}")
    print(f"Total time: {metrics.total_time_seconds:.4f}s")
    print(f"Average time per step: {metrics.avg_time_per_step:.4f}s")
    print(f"Min step time: {metrics.min_step_time:.4f}s")
    print(f"Max step time: {metrics.max_step_time:.4f}s")
    if metrics.steps_per_game:
        avg_steps = sum(metrics.steps_per_game) / len(metrics.steps_per_game)
        print(f"Average steps per game: {avg_steps:.1f}")
