# MCTS Configuration Constants

# Game Environment
STACK_SEQ_LEN = 40  # Max cards in dungeon deck (44 total - 4 dealt to room at start)
NUM_CARDS = 45  # Card embeddings: 0=pad, 1-44=actual cards
ACTION_SPACE = 5  # 0-3: Pick Card 1-4, 4: Run/Skip Room

# MCTS Parameters
MCTS_NUM_SIMULATIONS = 50000  # Number of simulations per move
MCTS_EXPLORATION_CONSTANT = 1.414  # UCB1 exploration constant (sqrt(2))
MCTS_MAX_DEPTH = 200  # Maximum depth for simulation rollout

# Evaluation Defaults
EVAL_NUM_GAMES = 100  # Number of games to play
EVAL_SAVE_INTERVAL = 100  # Save statistics every N games
EVAL_VERBOSE = True  # Print progress during evaluation

# Simulation Policy
USE_RANDOM_ROLLOUT = True  # Use random policy for rollouts (vs heuristic)

# Parallelization
MCTS_NUM_WORKERS = 8  # Number of parallel workers (0 or 1 = no parallelization)
# should be set to the number of cores available
# inflection point around 2000 simulations

# Test Configuration
MCTS_TEST_MAX_STEPS = MCTS_MAX_DEPTH  # Maximum number of steps to play in performance tests (safety limit)
MCTS_TEST_NUM_GAMES = 1  # Number of games to play in performance tests
MCTS_TEST_SEED = 42  # Default seed for reproducible performance tests

