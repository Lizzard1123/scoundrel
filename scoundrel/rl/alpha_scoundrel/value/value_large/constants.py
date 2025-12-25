STACK_SEQ_LEN = 40
NUM_CARDS = 45

# Network architecture
EMBED_DIM = 32          # Card and position embedding dimension
HIDDEN_DIM = 256        # Hidden layer dimension in value head

# Training hyperparameters
LR = 5e-4
BATCH_SIZE = 128
EPOCHS = 100
TRAIN_VAL_SPLIT = 0.9
MAX_GRAD_NORM = 1.0     # Maximum gradient norm for clipping

# Data
DEFAULT_MCTS_LOGS_DIR = "scoundrel/rl/mcts/logs/collected_games"
MAX_GAMES = None

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "value_large_epoch_"
CHECKPOINT_INTERVAL = 10
