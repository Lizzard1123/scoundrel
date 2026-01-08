STACK_SEQ_LEN = 40
NUM_CARDS = 45
ACTION_SPACE = 5

# Training hyperparameters (updated for advanced training)
LR = 5e-4                  # Higher LR for faster learning
BATCH_SIZE = 1024           # Smaller batches for stability
EPOCHS = 500               # Reduced from 1000, more efficient with advanced techniques
TRAIN_VAL_SPLIT = 0.9
MAX_GRAD_NORM = 1.0        # Gradient clipping
DROPOUT_RATE = 0.1         # Dropout for regularization

# Target distribution sharpening
# Temperature < 1.0 sharpens the MCTS visit distribution towards one-hot
# This makes training targets more decisive, improving action accuracy
TEMPERATURE = 0.5          # Sharpening temperature (0.5 = moderate sharpening)
USE_Q_WEIGHTS = False       # Weight visits by Q-values from MCTS

# Hybrid loss parameters
# Combines soft distribution matching with hard best-action classification
# hard_weight=0 is pure distribution matching, hard_weight=1 is pure classification
HARD_LOSS_WEIGHT = 1.0     # Weight for best-action classification loss

# Focal MSE parameters
# Adapts Focal Loss concept to regression by modulating MSE with error magnitude
# Higher gamma focuses training on hard examples (large prediction errors)
FOCAL_GAMMA = 2.0          # Gamma parameter for focal MSE: |y-ŷ|^γ * (y-ŷ)²

# Learning rate scheduling (warmup + cosine decay)
WARMUP_EPOCHS = 0          # No warmup (can be enabled if needed)
MIN_LR_RATIO = 1.0         # No decay (constant LR)

DEFAULT_MCTS_LOGS_DIR = "scoundrel/rl/mcts/logs/collected_games"
MAX_GAMES = None

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "policy_small_epoch_"
CHECKPOINT_INTERVAL = 5

