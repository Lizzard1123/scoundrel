"""
Alpha Scoundrel Value Large - Supervised learning from MCTS final scores.

This module implements a value network trained to predict expected game outcome
(final score) given a game state. The architecture is identical to PolicyLargeNet
but with a value head instead of an action head.
"""

from scoundrel.rl.alpha_scoundrel.value.value_large.network import ValueLargeNet
from scoundrel.rl.alpha_scoundrel.value.value_large.constants import (
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
    'ValueLargeNet',
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

