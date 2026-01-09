"""
Constants for AlphaGo Self-Play Training.

Uses REINFORCE policy gradients for true RL training:
- Win (+1): Reinforce actions taken
- Loss (-1): Discourage actions taken
"""

from pathlib import Path
from ..alphago_mcts.constants import (
    ALPHAGO_NUM_SIMULATIONS,
    ALPHAGO_NUM_WORKERS,
    ALPHAGO_PARALLEL_GAMES,
    EVAL_SEED,
    POLICY_LARGE_CHECKPOINT as ALPHAGO_POLICY_LARGE_CHECKPOINT,
    POLICY_SMALL_CHECKPOINT as ALPHAGO_POLICY_SMALL_CHECKPOINT,
    VALUE_LARGE_CHECKPOINT as ALPHAGO_VALUE_LARGE_CHECKPOINT,
)

# =============================================================================
# Self-play game generation
# =============================================================================
SELF_PLAY_GAMES_PER_ITERATION = 30  # Reduced for faster demonstration of colored logging
SELF_PLAY_NUM_WORKERS = ALPHAGO_NUM_WORKERS  # Internal MCTS workers (threading, not multiprocessing)
SELF_PLAY_SIMULATIONS = ALPHAGO_NUM_SIMULATIONS  # MCTS simulations per move during self-play
SELF_PLAY_USE_GPU = True               # Use GPU for inference during self-play
SELF_PLAY_PARALLEL_GAMES = ALPHAGO_PARALLEL_GAMES  # Number of games to generate simultaneously (each in separate process)

# =============================================================================
# Training configuration
# =============================================================================
POLICY_EPOCHS_PER_ITERATION = 20
VALUE_EPOCHS_PER_ITERATION = 20
BATCH_SIZE = 256

# =============================================================================
# Evaluation
# =============================================================================
EVAL_GAMES = 10  # Number of games in evaluation set
# EVAL_SEED imported from alphago_mcts.constants
EVAL_SIMULATIONS = 50  # Reduced for faster evaluation

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
POLICY_LARGE_CHECKPOINT = BASE_DIR / ALPHAGO_POLICY_LARGE_CHECKPOINT
POLICY_SMALL_CHECKPOINT = BASE_DIR / ALPHAGO_POLICY_SMALL_CHECKPOINT
VALUE_LARGE_CHECKPOINT = BASE_DIR / ALPHAGO_VALUE_LARGE_CHECKPOINT

# =============================================================================
# TensorBoard logging configuration
# =============================================================================
TENSORBOARD_LOG_INTERVAL = 10         # Log batch metrics every N batches
TENSORBOARD_HISTOGRAM_INTERVAL = 50   # Log weight histograms every N batches
LOG_WEIGHT_HISTOGRAMS = True          # Enable/disable weight histogram logging
LOG_GRADIENT_HISTOGRAMS = True        # Enable/disable gradient histogram logging
