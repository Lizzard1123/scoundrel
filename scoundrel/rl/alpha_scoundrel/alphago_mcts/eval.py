"""
Console script for evaluating AlphaGo-style MCTS agent.

This evaluates the AlphaGo MCTS agent on win percentage and average score.
Optionally compares with vanilla MCTS baseline.
"""
import argparse

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_agent import AlphaGoAgent
from scoundrel.rl.alpha_scoundrel.alphago_mcts.constants import (
    ALPHAGO_NUM_SIMULATIONS,
    ALPHAGO_C_PUCT,
    ALPHAGO_VALUE_WEIGHT,
    ALPHAGO_MAX_DEPTH,
    ALPHAGO_NUM_WORKERS,
    EVAL_SEED,
    EVAL_NUM_GAMES,
)


def run_evaluation(
    agent: AlphaGoAgent,
    num_games: int = EVAL_NUM_GAMES,
    seed: int = EVAL_SEED,
    verbose: bool = False
) -> dict:
    """
    Run AlphaGo MCTS evaluation.
    
    Args:
        agent: The AlphaGoAgent instance to evaluate.
        num_games: Number of games to play
        seed: Base seed for GameManager
        verbose: Print progress during evaluation
        
    Returns:
        Dictionary with:
        - num_games, wins, win_percentage
        - average_score, best_score, worst_score
        - total_score, scores list
    """
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
            cache_stats = agent.get_cache_stats()
            print(f"  Game {game_num + 1} finished: Score={score}, {'Win' if score > 0 else 'Loss'}")
            print(f"    Cache hit rate: {cache_stats['hit_rate']:.2%}")
    
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
    """Print evaluation results in formatted output."""
    print(f"\n=== AlphaGo MCTS Evaluation Results ===")
    print(f"Configuration:")
    print(f"  Simulations: {ALPHAGO_NUM_SIMULATIONS}")
    print(f"  C_PUCT: {ALPHAGO_C_PUCT}")
    print(f"  Value Weight (λ): {ALPHAGO_VALUE_WEIGHT}")
    print(f"  Max Depth: {ALPHAGO_MAX_DEPTH}")
    print(f"  Workers: {ALPHAGO_NUM_WORKERS}")
    print(f"\nResults:")
    print(f"  Games Played: {results['num_games']}")
    print(f"  Wins: {results['wins']}")
    print(f"  Win Percentage: {results['win_percentage']:.2f}%")
    print(f"  Average Score: {results['average_score']:.2f}")
    print(f"  Best Score: {results['best_score']}")
    print(f"  Worst Score: {results['worst_score']}")
    print(f"  Total Score: {results['total_score']}")


def compare_with_baseline(alphago_results: dict, mcts_results: dict):
    """
    Compare AlphaGo MCTS with vanilla MCTS.
    
    Shows:
    - Win rate improvement
    - Average score improvement
    - Simulation efficiency (score per simulation)
    
    Args:
        alphago_results: Results from AlphaGo MCTS evaluation
        mcts_results: Results from vanilla MCTS evaluation
    """
    print(f"\n=== Comparison: AlphaGo MCTS vs Vanilla MCTS ===")
    
    # Win rate comparison
    alphago_win_rate = alphago_results['win_percentage']
    mcts_win_rate = mcts_results['win_percentage']
    win_rate_diff = alphago_win_rate - mcts_win_rate
    
    print(f"\nWin Rate:")
    print(f"  AlphaGo MCTS: {alphago_win_rate:.2f}%")
    print(f"  Vanilla MCTS: {mcts_win_rate:.2f}%")
    print(f"  Difference: {win_rate_diff:+.2f}%")
    
    # Average score comparison
    alphago_avg_score = alphago_results['average_score']
    mcts_avg_score = mcts_results['average_score']
    score_diff = alphago_avg_score - mcts_avg_score
    
    print(f"\nAverage Score:")
    print(f"  AlphaGo MCTS: {alphago_avg_score:.2f}")
    print(f"  Vanilla MCTS: {mcts_avg_score:.2f}")
    print(f"  Difference: {score_diff:+.2f}")
    
    # Simulation efficiency
    print(f"\nSimulation Efficiency:")
    print(f"  AlphaGo MCTS: {ALPHAGO_NUM_SIMULATIONS} simulations")
    print(f"  Vanilla MCTS: (specify in vanilla MCTS constants)")
    
    # Overall assessment
    if win_rate_diff > 0 or score_diff > 0:
        print(f"\n✓ AlphaGo MCTS outperforms vanilla MCTS!")
    else:
        print(f"\n✗ AlphaGo MCTS underperforms vanilla MCTS.")


def main():
    """Console script entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate AlphaGo-style MCTS agent on win percentage and average score."
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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with vanilla MCTS baseline (requires running vanilla MCTS eval first)"
    )
    
    args = parser.parse_args()
    
    # Create a default agent for standalone evaluation
    agent = AlphaGoAgent()
    
    results = run_evaluation(
        agent=agent,
        num_games=args.num_games,
        seed=args.seed,
        verbose=args.verbose
    )
    print_results(results)
    
    if args.compare:
        print("\nNote: Baseline comparison requires vanilla MCTS results.")
        print("Run this script and vanilla MCTS eval separately, then compare manually.")


if __name__ == "__main__":
    main()

