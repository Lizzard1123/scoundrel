"""
Alpha Scoundrel Policy Small - Supervised learning from MCTS visit distributions.

This module implements a small policy network trained to match MCTS visit distributions
from collected game logs. Uses a single MLP with regular game state input and
dungeon stack sums instead of transformer sequence encoding.
"""

from scoundrel.rl.alpha_scoundrel.policy.policy_small.network import PolicySmallNet
from scoundrel.rl.alpha_scoundrel.policy.policy_small.constants import (
    ACTION_SPACE,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PREFIX,
    DEFAULT_MCTS_LOGS_DIR,
    EPOCHS,
    HIDDEN_DIM,
    LR,
    MAX_GAMES,
    SCALAR_ENCODER_OUT,
    STACK_SEQ_LEN,
    TRAIN_VAL_SPLIT,
)

__all__ = [
    'PolicySmallNet',
    'ACTION_SPACE',
    'BATCH_SIZE',
    'CHECKPOINT_DIR',
    'CHECKPOINT_INTERVAL',
    'CHECKPOINT_PREFIX',
    'DEFAULT_MCTS_LOGS_DIR',
    'EPOCHS',
    'HIDDEN_DIM',
    'LR',
    'MAX_GAMES',
    'SCALAR_ENCODER_OUT',
    'STACK_SEQ_LEN',
    'TRAIN_VAL_SPLIT',
]

