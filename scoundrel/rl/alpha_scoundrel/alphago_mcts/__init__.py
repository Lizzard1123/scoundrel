"""
AlphaGo-style MCTS for Scoundrel.

Combines neural networks (PolicyLarge, PolicySmall, ValueLarge) with MCTS
using PUCT formula and hybrid evaluation.
"""

from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_agent import AlphaGoAgent
from scoundrel.rl.alpha_scoundrel.alphago_mcts.alphago_node import AlphaGoNode
from scoundrel.rl.alpha_scoundrel.alphago_mcts.constants import (
    ALPHAGO_NUM_SIMULATIONS,
    ALPHAGO_C_PUCT,
    ALPHAGO_VALUE_WEIGHT,
    ALPHAGO_MAX_DEPTH,
    ALPHAGO_NUM_WORKERS,
)

__all__ = [
    'AlphaGoAgent',
    'AlphaGoNode',
    'ALPHAGO_NUM_SIMULATIONS',
    'ALPHAGO_C_PUCT',
    'ALPHAGO_VALUE_WEIGHT',
    'ALPHAGO_MAX_DEPTH',
    'ALPHAGO_NUM_WORKERS',
]

