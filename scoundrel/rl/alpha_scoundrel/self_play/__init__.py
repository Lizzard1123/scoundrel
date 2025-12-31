"""
AlphaGo Self-Play Training Package.

Implements iterative self-play training with REINFORCE policy gradients:
- Generate self-play games using current best AlphaGoAgent
- Train policy with REINFORCE: Win (+1) reinforces, Loss (-1) discourages
- Train value network on game outcomes
- Evaluate and update best checkpoints
"""

from .constants import (
    SELF_PLAY_GAMES_PER_ITERATION,
    SELF_PLAY_NUM_WORKERS,
    SELF_PLAY_SIMULATIONS,
    POLICY_EPOCHS_PER_ITERATION,
    VALUE_EPOCHS_PER_ITERATION,
    BATCH_SIZE,
    EVAL_GAMES,
    EVAL_SEED,
    EVAL_SIMULATIONS,
    CHECKPOINT_BASE_DIR,
    RUNS_BASE_DIR,
    POLICY_LR,
    POLICY_MAX_GRAD_NORM,
    ENTROPY_COEF,
    REWARD_TYPE,
    VALUE_LR,
    VALUE_MAX_GRAD_NORM,
    TRAIN_VAL_SPLIT,
    STACK_SEQ_LEN,
    ITERATION_PREFIX,
    POLICY_CHECKPOINT_NAME,
    VALUE_CHECKPOINT_NAME,
    BEST_CHECKPOINT_DIR,
    POLICY_LARGE_CHECKPOINT,
    POLICY_SMALL_CHECKPOINT,
    VALUE_LARGE_CHECKPOINT,
)

__version__ = "0.2.0"
