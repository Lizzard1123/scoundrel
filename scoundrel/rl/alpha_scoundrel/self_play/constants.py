"""
Constants for AlphaGo Self-Play Training.

Uses REINFORCE policy gradients for true RL training:
- Win (+1): Reinforce actions taken
- Loss (-1): Discourage actions taken
"""

from pathlib import Path

# =============================================================================
# Self-play game generation
# =============================================================================
SELF_PLAY_GAMES_PER_ITERATION = 50
SELF_PLAY_NUM_WORKERS = 8              # Internal MCTS workers (threading, not multiprocessing)
SELF_PLAY_SIMULATIONS = 800            # Fewer than eval (3000), more than training collection (300)
SELF_PLAY_USE_GPU = True               # Use GPU for inference during self-play

# =============================================================================
# Training configuration
# =============================================================================
POLICY_EPOCHS_PER_ITERATION = 20
VALUE_EPOCHS_PER_ITERATION = 20
BATCH_SIZE = 256

# =============================================================================
# Evaluation
# =============================================================================
EVAL_GAMES = 50
EVAL_SEED = 42
EVAL_SIMULATIONS = 1600  # Full evaluation strength

# =============================================================================
# Checkpointing and directories
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
SELF_PLAY_BASE_DIR = BASE_DIR / "self_play"
CHECKPOINT_BASE_DIR = SELF_PLAY_BASE_DIR / "checkpoints"
RUNS_BASE_DIR = SELF_PLAY_BASE_DIR / "runs"

# Checkpoint naming
ITERATION_PREFIX = "iteration_"
POLICY_CHECKPOINT_NAME = "policy_large_best.pt"
VALUE_CHECKPOINT_NAME = "value_large_best.pt"
BEST_CHECKPOINT_DIR = "best"

# =============================================================================
# Policy network training (REINFORCE)
# =============================================================================
POLICY_LR = 1e-4                 # Lower LR for fine-tuning
POLICY_MAX_GRAD_NORM = 1.0
ENTROPY_COEF = 0.01              # Entropy bonus for exploration
REWARD_TYPE = "binary"           # "binary" (+1/-1), "normalized", or "scaled"

# =============================================================================
# Value network training
# =============================================================================
VALUE_LR = 1e-4                  # Lower LR for fine-tuning
VALUE_MAX_GRAD_NORM = 1.0

# =============================================================================
# Data loading
# =============================================================================
TRAIN_VAL_SPLIT = 0.9
STACK_SEQ_LEN = 40  # Sequence length for dungeon card encoding

# =============================================================================
# Default initial checkpoints (can be overridden via CLI)
# =============================================================================
POLICY_LARGE_CHECKPOINT = BASE_DIR / "policy" / "policy_large" / "checkpoints" / "run_20251231_093510" / "policy_large_epoch_60.pt"
POLICY_SMALL_CHECKPOINT = BASE_DIR / "policy" / "policy_small" / "checkpoints" / "run_20251229_184938" / "policy_small_epoch_10.pt"
VALUE_LARGE_CHECKPOINT = BASE_DIR / "value" / "value_large" / "checkpoints" / "100e_193mse" / "value_large_epoch_100.pt"

# =============================================================================
# TensorBoard logging configuration
# =============================================================================
TENSORBOARD_LOG_INTERVAL = 10         # Log batch metrics every N batches
TENSORBOARD_HISTOGRAM_INTERVAL = 50   # Log weight histograms every N batches
LOG_WEIGHT_HISTOGRAMS = True          # Enable/disable weight histogram logging
LOG_GRADIENT_HISTOGRAMS = True        # Enable/disable gradient histogram logging
