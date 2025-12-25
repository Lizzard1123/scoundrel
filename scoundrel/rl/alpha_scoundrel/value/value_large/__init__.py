"""
Alpha Scoundrel Value Large - Supervised learning from MCTS final scores.

This module implements a value network trained to predict expected game outcome
(final score) given a game state. The architecture uses position-aware encoding
for known dungeon cards and aggregate statistics for unknown cards.
"""

from scoundrel.rl.alpha_scoundrel.value.value_large.network import ValueLargeNet
from scoundrel.rl.alpha_scoundrel.value.value_large.constants import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PREFIX,
    DEFAULT_MCTS_LOGS_DIR,
    EMBED_DIM,
    EPOCHS,
    HIDDEN_DIM,
    LR,
    MAX_GRAD_NORM,
    MAX_GAMES,
    NUM_CARDS,
    STACK_SEQ_LEN,
    TRAIN_VAL_SPLIT,
)

__all__ = [
    'ValueLargeNet',
    'BATCH_SIZE',
    'CHECKPOINT_DIR',
    'CHECKPOINT_INTERVAL',
    'CHECKPOINT_PREFIX',
    'DEFAULT_MCTS_LOGS_DIR',
    'EMBED_DIM',
    'EPOCHS',
    'HIDDEN_DIM',
    'LR',
    'MAX_GRAD_NORM',
    'MAX_GAMES',
    'NUM_CARDS',
    'STACK_SEQ_LEN',
    'TRAIN_VAL_SPLIT',
]
