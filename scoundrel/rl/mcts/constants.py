STACK_SEQ_LEN = 40  # Max cards in dungeon deck (44 total - 4 dealt to room at start)
NUM_CARDS = 45  # Card embeddings: 0=pad, 1-44=actual cards
ACTION_SPACE = 5  # 0-3: Pick Card 1-4, 4: Run/Skip Room

MCTS_NUM_SIMULATIONS = 1000000
MCTS_EXPLORATION_CONSTANT = 1.414  # UCB1 exploration constant (sqrt(2))
MCTS_MAX_DEPTH = 120  # Maximum depth for simulation rollout (games typically end <100 steps, 1.5x safety margin)

EVAL_NUM_GAMES = 10
EVAL_SEED = 42

USE_RANDOM_ROLLOUT = True  # Use random policy for rollouts (vs heuristic)

MCTS_NUM_WORKERS = 8  # Number of parallel workers (0 or 1 = no parallelization)

MCTS_TRANSPOSITION_TABLE_SIZE = 100000  # Maximum number of cached state evaluations (LRU eviction)

MCTS_TEST_MAX_STEPS = MCTS_MAX_DEPTH
MCTS_TEST_NUM_GAMES = 1
MCTS_TEST_SEED = 42

