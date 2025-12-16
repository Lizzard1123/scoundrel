"""
MCTS Evaluation Script for Scoundrel.
"""
import argparse
from pathlib import Path
from collections import deque
import json
import time

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.mcts.mcts_agent import MCTSAgent
from scoundrel.rl.mcts.constants import (
    MCTS_NUM_SIMULATIONS,
    MCTS_NUM_WORKERS,
    EVAL_NUM_GAMES,
    EVAL_SAVE_INTERVAL,
    EVAL_VERBOSE,
)


def _default_paths(base_dir: Path):
    """Create default paths for logs and statistics."""
    logs_dir = base_dir / "logs"
    stats_file = logs_dir / "mcts_stats.json"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir, stats_file


class MCTSStatistics:
    """Track and store MCTS performance statistics."""
    
    def __init__(self):
        self.games_played = 0
        self.total_score = 0
        self.best_score = float('-inf')
        self.worst_score = float('inf')
        self.wins = 0  # Games where health > 0
        self.scores = []
        self.avg_scores = []  # Moving average
        self.best_scores = []  # Best score so far
    
    def update(self, score: int):
        """Update statistics with a new game result."""
        self.games_played += 1
        self.total_score += score
        self.scores.append(score)
        
        if score > self.best_score:
            self.best_score = score
        if score < self.worst_score:
            self.worst_score = score
        if score > 0:
            self.wins += 1
        
        # Track moving averages and best scores
        self.avg_scores.append(self.total_score / self.games_played)
        self.best_scores.append(self.best_score)
    
    def get_stats(self):
        """Return current statistics as a dictionary."""
        return {
            "games_played": self.games_played,
            "avg_score": self.total_score / max(1, self.games_played),
            "best_score": self.best_score,
            "worst_score": self.worst_score,
            "win_rate": self.wins / max(1, self.games_played),
            "recent_avg": sum(self.scores[-100:]) / min(100, len(self.scores)),
        }
    
    def save_to_file(self, filepath: Path):
        """Save statistics to JSON file."""
        data = {
            "stats": self.get_stats(),
            "history": {
                "scores": self.scores,
                "avg_scores": self.avg_scores,
                "best_scores": self.best_scores,
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Path):
        """Load statistics from JSON file."""
        stats = cls()
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            stats.scores = data["history"]["scores"]
            stats.avg_scores = data["history"]["avg_scores"]
            stats.best_scores = data["history"]["best_scores"]
            stats.games_played = len(stats.scores)
            stats.total_score = sum(stats.scores)
            stats.best_score = max(stats.scores) if stats.scores else float('-inf')
            stats.worst_score = min(stats.scores) if stats.scores else float('inf')
            stats.wins = sum(1 for s in stats.scores if s > 0)
        return stats


def play_mcts_game(
    agent: MCTSAgent,
    engine: GameManager,
    verbose: bool = False
) -> int:
    """
    Play a single game using MCTS agent.
    
    Args:
        agent: MCTS agent
        engine: Game engine
        verbose: Whether to print game progress
        
    Returns:
        Final game score
    """
    state = engine.restart()
    turns = 0
    
    while not state.game_over:
        # Get action from MCTS
        action_idx = agent.select_action(state)
        action_enum = agent.translator.decode_action(action_idx)
        
        # Execute action
        engine.execute_turn(action_enum)
        state = engine.get_state()
        turns += 1
        
        if verbose and turns % 10 == 0:
            print(f"  Turn {turns}: Health={state.health}, Score={state.score}")
    
    return state.score


def evaluate_mcts(
    num_simulations: int = MCTS_NUM_SIMULATIONS,
    num_games: int = EVAL_NUM_GAMES,
    save_interval: int = EVAL_SAVE_INTERVAL,
    verbose: bool = EVAL_VERBOSE,
    resume_from: str = None,
    num_workers: int = MCTS_NUM_WORKERS,
):
    """
    Evaluate MCTS agent by playing multiple games and collecting statistics.
    
    Args:
        num_simulations: Number of MCTS simulations per move
        num_games: Number of games to play
        save_interval: Save statistics every N games
        verbose: Print progress
        resume_from: Path to resume statistics from
        num_workers: Number of parallel workers for MCTS (0 or 1 disables parallelization)
    """
    base_dir = Path(__file__).parent
    logs_dir, stats_file = _default_paths(base_dir)
    
    print(f"--- Starting MCTS Evaluation ---")
    print(f"Simulations per move: {num_simulations}")
    print(f"Number of games: {num_games}")
    print(f"Number of workers: {num_workers}")
    print(f"Parallelization: {'Enabled' if num_workers > 1 else 'Disabled'}")
    print(f"Logs directory: {logs_dir}")
    
    # Initialize agent and engine
    agent = MCTSAgent(num_simulations=num_simulations, num_workers=num_workers)
    engine = GameManager()
    
    # Load or create statistics
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            stats = MCTSStatistics.load_from_file(resume_path)
            print(f"Resumed from: {resume_path}")
            print(f"Previous games: {stats.games_played}")
        else:
            stats = MCTSStatistics()
            print(f"Resume file not found, starting fresh")
    else:
        stats = MCTSStatistics()
    
    # Track timing
    start_time = time.time()
    window = deque(maxlen=save_interval)
    
    # Play games
    for i in range(1, num_games + 1):
        game_start = time.time()
        
        if verbose:
            print(f"\n[Game {i}/{num_games}]")
        
        score = play_mcts_game(agent, engine, verbose=verbose)
        stats.update(score)
        window.append(score)
        
        game_time = time.time() - game_start
        
        if verbose:
            print(f"  Final Score: {score}")
            print(f"  Time: {game_time:.2f}s")
        
        # Periodic reporting and saving
        if i % save_interval == 0:
            elapsed = time.time() - start_time
            current_stats = stats.get_stats()
            window_avg = sum(window) / len(window)
            
            print(f"\n=== Progress Report (Game {i}/{num_games}) ===")
            print(f"Average Score: {current_stats['avg_score']:.2f}")
            print(f"Recent Avg ({save_interval} games): {window_avg:.2f}")
            print(f"Best Score: {current_stats['best_score']}")
            print(f"Win Rate: {current_stats['win_rate']:.2%}")
            print(f"Time Elapsed: {elapsed:.1f}s")
            print(f"Time per Game: {elapsed/i:.2f}s")
            
            # Save statistics
            stats.save_to_file(stats_file)
            print(f"Statistics saved to: {stats_file}")
    
    # Final summary
    print(f"\n=== Final Results ===")
    final_stats = stats.get_stats()
    print(f"Total Games: {stats.games_played}")
    print(f"Average Score: {final_stats['avg_score']:.2f}")
    print(f"Best Score: {final_stats['best_score']}")
    print(f"Worst Score: {final_stats['worst_score']}")
    print(f"Win Rate: {final_stats['win_rate']:.2%}")
    print(f"Total Time: {time.time() - start_time:.1f}s")
    
    # Save final statistics
    stats.save_to_file(stats_file)
    print(f"\nFinal statistics saved to: {stats_file}")
    
    return stats


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MCTS agent on Scoundrel."
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=MCTS_NUM_SIMULATIONS,
        help="Number of MCTS simulations per move"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=EVAL_NUM_GAMES,
        help="Number of games to play"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=MCTS_NUM_WORKERS,
        help="Number of parallel workers (0 or 1 disables parallelization)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=EVAL_SAVE_INTERVAL,
        help="Save statistics every N games"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to statistics file to resume from"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_mcts(
        num_simulations=args.num_simulations,
        num_games=args.num_games,
        save_interval=args.save_interval,
        verbose=not args.quiet,
        resume_from=args.resume_from,
        num_workers=args.num_workers,
    )



