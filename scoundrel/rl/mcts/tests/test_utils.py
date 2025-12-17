"""
Shared utilities for MCTS performance tests.
"""
import time
from typing import List
from dataclasses import dataclass

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.mcts.mcts_agent import MCTSAgent


@dataclass
class StepTiming:
    """Timing information for a single game step."""
    step_number: int
    time_seconds: float
    action_taken: int


@dataclass
class PerformanceMetrics:
    """Performance metrics collected from a test run."""
    total_steps: int
    total_time_seconds: float
    avg_time_per_step: float
    min_step_time: float
    max_step_time: float
    steps_per_game: List[int]
    step_timings: List[StepTiming]
    num_games: int
    num_workers: int
    num_simulations: int


def play_games_with_timing(
    agent: MCTSAgent,
    engine: GameManager,
    num_games: int,
    game_seed: int
) -> PerformanceMetrics:
    """
    Play a fixed number of games and measure time for each step.
    
    Each game uses a deterministic seed based on game_seed + game_number.
    
    Args:
        agent: MCTS agent to use
        engine: Game manager instance (will be recreated per game)
        num_games: Number of games to play
        game_seed: Base seed for GameManager (deterministic deck shuffling)
        
    Returns:
        PerformanceMetrics with timing information
    """
    step_timings: List[StepTiming] = []
    steps_per_game: List[int] = []
    
    total_start = time.perf_counter()
    
    for game_num in range(num_games):
        # Create new game with deterministic seed
        engine_seed = game_seed + game_num
        engine = GameManager(seed=engine_seed)
        state = engine.restart()
        
        current_game_steps = 0
        step_number = len(step_timings)
        
        while not state.game_over:
            # Measure time for action selection
            step_start = time.perf_counter()
            
            # Get action from MCTS
            action_idx = agent.select_action(state)
            action_enum = agent.translator.decode_action(action_idx)
            
            # Execute action
            engine.execute_turn(action_enum)
            state = engine.get_state()
            
            step_time = time.perf_counter() - step_start
            
            step_number += 1
            current_game_steps += 1
            
            step_timings.append(StepTiming(
                step_number=step_number,
                time_seconds=step_time,
                action_taken=action_idx
            ))
        
        steps_per_game.append(current_game_steps)
    
    total_time = time.perf_counter() - total_start
    
    # Calculate statistics
    step_times = [s.time_seconds for s in step_timings]
    avg_time = sum(step_times) / len(step_times) if step_times else 0.0
    min_time = min(step_times) if step_times else 0.0
    max_time = max(step_times) if step_times else 0.0
    
    return PerformanceMetrics(
        total_steps=len(step_timings),
        total_time_seconds=total_time,
        avg_time_per_step=avg_time,
        min_step_time=min_time,
        max_step_time=max_time,
        steps_per_game=steps_per_game,
        step_timings=step_timings,
        num_games=len(steps_per_game),
        num_workers=agent.num_workers,
        num_simulations=agent.num_simulations
    )
