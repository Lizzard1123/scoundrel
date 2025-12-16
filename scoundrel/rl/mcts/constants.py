# MCTS Configuration Constants

# Game Environment
STACK_SEQ_LEN = 40  # Max cards in dungeon deck (44 total - 4 dealt to room at start)
NUM_CARDS = 45  # Card embeddings: 0=pad, 1-44=actual cards
ACTION_SPACE = 5  # 0-3: Pick Card 1-4, 4: Run/Skip Room

# MCTS Parameters
MCTS_NUM_SIMULATIONS = 200  # Number of simulations per move
MCTS_EXPLORATION_CONSTANT = 1.414  # UCB1 exploration constant (sqrt(2))
MCTS_MAX_DEPTH = 200  # Maximum depth for simulation rollout

# Evaluation Defaults
EVAL_NUM_GAMES = 1000  # Number of games to play
EVAL_SAVE_INTERVAL = 100  # Save statistics every N games
EVAL_VERBOSE = True  # Print progress during evaluation

# Simulation Policy
# Random rollout: Faster, more exploration, works well with many simulations
# Heuristic rollout: Slower, better per-simulation, works with fewer simulations
USE_RANDOM_ROLLOUT = True  # Use random policy for rollouts (vs heuristic)

# Recommended configurations:
# - High simulations (500+): USE_RANDOM_ROLLOUT = True (volume compensates)
# - Low simulations (50-200): USE_RANDOM_ROLLOUT = False (quality matters more)
# - Interactive play: USE_RANDOM_ROLLOUT = False (better per-move quality)

