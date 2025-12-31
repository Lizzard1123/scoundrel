"""
Constants for Policy Large Transformer Network.

Architecture: Transformer-based with cross-attention between room and dungeon.
"""

STACK_SEQ_LEN = 40
NUM_CARDS = 45
ACTION_SPACE = 5

# Transformer architecture
EMBED_DIM = 128            # Embedding dimension for cards and features
HIDDEN_DIM = 512           # Hidden dimension in policy head
NUM_HEADS = 8              # Number of attention heads (must divide EMBED_DIM)
NUM_TRANSFORMER_LAYERS = 3 # Layers in room/dungeon encoders
FF_DIM_MULTIPLIER = 4.0    # FFN hidden dim = EMBED_DIM * this
DROPOUT_RATE = 0.1         # Dropout rate

# Legacy compatibility (unused but kept for checkpoint loading)
SCALAR_ENCODER_OUT = 512

# Training hyperparameters
LR = 1e-3                  # Higher LR for faster learning
BATCH_SIZE = 256           # Smaller batches for transformer memory
EPOCHS = 500
TRAIN_VAL_SPLIT = 0.9
MAX_GRAD_NORM = 1.0        # Gradient clipping

# Data
DEFAULT_MCTS_LOGS_DIR = "scoundrel/rl/mcts/logs/collected_games"
MAX_GAMES = None

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "policy_large_epoch_"
CHECKPOINT_INTERVAL = 10

# Learning rate scheduling (disabled - using constant LR)
WARMUP_EPOCHS = 0          # No warmup
MIN_LR_RATIO = 1.0         # No decay (constant LR)
