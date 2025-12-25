"""
Alpha Scoundrel Policy Large - Supervised learning from MCTS visit distributions.

This module implements a policy network trained to match MCTS visit distributions
from collected game logs. The architecture is identical to transformer_mlp but
with only an actor head (no critic).
"""

from scoundrel.rl.alpha_scoundrel.policy.policy_large.network import PolicyLargeNet
from scoundrel.rl.alpha_scoundrel.policy.policy_large.constants import (
    ACTION_SPACE,
    BATCH_SIZE,
    CARD_EMBEDDING_DIM,
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PREFIX,
    DEFAULT_MCTS_LOGS_DIR,
    EPOCHS,
    HIDDEN_DIM,
    LR,
    MAX_GAMES,
    NUM_CARDS,
    SCALAR_ENCODER_OUT,
    STACK_SEQ_LEN,
    TRAIN_VAL_SPLIT,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NLAYERS,
)

__all__ = [
    'PolicyLargeNet',
    'ACTION_SPACE',
    'BATCH_SIZE',
    'CARD_EMBEDDING_DIM',
    'CHECKPOINT_DIR',
    'CHECKPOINT_INTERVAL',
    'CHECKPOINT_PREFIX',
    'DEFAULT_MCTS_LOGS_DIR',
    'EPOCHS',
    'HIDDEN_DIM',
    'LR',
    'MAX_GAMES',
    'NUM_CARDS',
    'SCALAR_ENCODER_OUT',
    'STACK_SEQ_LEN',
    'TRAIN_VAL_SPLIT',
    'TRANSFORMER_NHEAD',
    'TRANSFORMER_NLAYERS',
]

