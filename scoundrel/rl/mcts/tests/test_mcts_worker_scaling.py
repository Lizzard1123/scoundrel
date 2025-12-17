"""
Pytest tests for MCTS worker scaling performance.

These tests measure performance scaling with different numbers of workers.
"""
import pytest

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.mcts.mcts_agent import MCTSAgent
from scoundrel.rl.mcts.constants import (
    MCTS_NUM_SIMULATIONS,
    MCTS_EXPLORATION_CONSTANT,
    MCTS_MAX_DEPTH,
    USE_RANDOM_ROLLOUT,
    MCTS_TEST_NUM_GAMES,
)
from .test_utils import play_games_with_timing


@pytest.mark.parametrize("num_workers", [1, 4, 6, 8, 10, 12])
def test_mcts_worker_scaling(game_seed, num_workers):
    """
    Test MCTS performance with different numbers of workers.
    
    Parametrized test to measure performance scaling with worker count.
    Note: Different worker counts will produce different game trajectories
    due to the nature of root parallelization in MCTS.
    """
    agent = MCTSAgent(
        num_simulations=MCTS_NUM_SIMULATIONS,
        num_workers=num_workers,
        exploration_constant=MCTS_EXPLORATION_CONSTANT,
        max_depth=MCTS_MAX_DEPTH,
        use_random_rollout=USE_RANDOM_ROLLOUT
    )
    
    engine = GameManager(seed=game_seed)
    metrics = play_games_with_timing(
        agent=agent,
        engine=engine,
        num_games=MCTS_TEST_NUM_GAMES,
        game_seed=game_seed
    )
    
    assert metrics.total_steps > 0
    assert metrics.num_workers == num_workers
    assert metrics.num_games == MCTS_TEST_NUM_GAMES
    
    # Get cache statistics
    cache_stats = agent.get_cache_stats()
    
    print(f"\n=== Worker Scaling Test ({num_workers} workers) ===")
    print(f"Games played: {metrics.num_games}")
    print(f"Total steps: {metrics.total_steps}")
    print(f"Avg time per step: {metrics.avg_time_per_step:.4f}s")
    print(f"Total time: {metrics.total_time_seconds:.4f}s")
    print(f"\nCache Statistics:")
    print(f"  Cache hits: {cache_stats['hits']}")
    print(f"  Cache misses: {cache_stats['misses']}")
    print(f"  Cache size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
