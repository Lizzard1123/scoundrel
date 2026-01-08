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

# Advanced training parameters
FOCAL_GAMMA = 2.0       # Gamma parameter for focal MSE loss
WARMUP_EPOCHS = 0       # Number of warmup epochs for learning rate
MIN_LR_RATIO = 0.1      # Minimum LR as ratio of initial LR (for cosine decay)

# Data
DEFAULT_MCTS_LOGS_DIR = "scoundrel/rl/mcts/logs/collected_games"
MAX_GAMES = None

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "value_large_epoch_"
CHECKPOINT_INTERVAL = 10
