STACK_SEQ_LEN = 40
NUM_CARDS = 45
ACTION_SPACE = 5

# Network architecture
EMBED_DIM = 64         # Card and position embedding dimension
SCALAR_ENCODER_OUT = 2048  # Scalar encoder output dimension (from policy small)
HIDDEN_DIM = 1024      # Hidden layer dimension in policy head (from policy small)

# Training hyperparameters
LR = 1e-3
BATCH_SIZE = 256
EPOCHS = 1000
TRAIN_VAL_SPLIT = 0.9
MAX_GRAD_NORM = 1.0     # Maximum gradient norm for clipping

# Data
DEFAULT_MCTS_LOGS_DIR = "scoundrel/rl/mcts/logs/collected_games"
MAX_GAMES = None

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "policy_large_epoch_"
CHECKPOINT_INTERVAL = 10
