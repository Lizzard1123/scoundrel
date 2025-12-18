"""
Console script for evaluating MCTS parameters on win percentage and average score.

This evaluates the current MCTS parameters from constants.py.
"""
import argparse

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.mcts.mcts_agent import MCTSAgent
from scoundrel.rl.mcts.constants import (
    MCTS_NUM_SIMULATIONS,
    MCTS_EXPLORATION_CONSTANT,
    MCTS_MAX_DEPTH,
    MCTS_NUM_WORKERS,
    EVAL_SEED,
    EVAL_NUM_GAMES,
    USE_RANDOM_ROLLOUT,
)


def run_evaluation(num_games: int = EVAL_NUM_GAMES, seed: int = EVAL_SEED, verbose: bool = False):
    """
    Run MCTS evaluation and return results.
    
    Args:
        num_games: Number of games to play
        seed: Base seed for GameManager
        verbose: Print progress during evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    agent = MCTSAgent(
        num_simulations=MCTS_NUM_SIMULATIONS,
        exploration_constant=MCTS_EXPLORATION_CONSTANT,
        max_depth=MCTS_MAX_DEPTH,
        num_workers=MCTS_NUM_WORKERS,
        use_random_rollout=USE_RANDOM_ROLLOUT,
    )
    
    scores = []
    wins = 0
    
    for game_num in range(num_games):
        engine_seed = seed + game_num
        
        if verbose:
            print(f"Playing game {game_num + 1}/{num_games} (seed={engine_seed})...")
        
        engine = GameManager(seed=engine_seed)
        state = engine.restart()
        
        while not state.game_over:
            action_idx = agent.select_action(state)
            action_enum = agent.translator.decode_action(action_idx)
            
            engine.execute_turn(action_enum)
            state = engine.get_state()
        
        score = state.score
        scores.append(score)
        
        if score > 0:
            wins += 1
        
        if verbose:
            print(f"  Game {game_num + 1} finished: Score={score}, {'Win' if score > 0 else 'Loss'}")
    
    total_score = sum(scores)
    win_percentage = (wins / num_games) * 100.0 if num_games > 0 else 0.0
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


def print_results(results: dict):
    """Print evaluation results."""
    print(f"\n=== MCTS Evaluation Results ===")
    print(f"Configuration:")
    print(f"  Simulations: {MCTS_NUM_SIMULATIONS}")
    print(f"  Exploration Constant: {MCTS_EXPLORATION_CONSTANT}")
    print(f"  Max Depth: {MCTS_MAX_DEPTH}")
    print(f"  Workers: {MCTS_NUM_WORKERS}")
    print(f"\nResults:")
    print(f"  Games Played: {results['num_games']}")
    print(f"  Wins: {results['wins']}")
    print(f"  Win Percentage: {results['win_percentage']:.2f}%")
    print(f"  Average Score: {results['average_score']:.2f}")
    print(f"  Best Score: {results['best_score']}")
    print(f"  Worst Score: {results['worst_score']}")
    print(f"  Total Score: {results['total_score']}")


def main():
    """Console script entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate MCTS parameters on win percentage and average score."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress during evaluation"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=EVAL_NUM_GAMES,
        help=f"Number of games to play (default: {EVAL_NUM_GAMES})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=EVAL_SEED,
        help=f"Base seed for GameManager (default: {EVAL_SEED})"
    )
    
    args = parser.parse_args()
    
    results = run_evaluation(
        num_games=args.num_games,
        seed=args.seed,
        verbose=args.verbose
    )
    print_results(results)


if __name__ == "__main__":
    main()
