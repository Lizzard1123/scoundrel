"""
Pytest configuration and shared fixtures for MCTS performance tests.
"""
import pytest
from scoundrel.rl.mcts.constants import MCTS_TEST_SEED


@pytest.fixture
def game_seed():
    """Seed for GameManager (deterministic deck shuffling)."""
    return MCTS_TEST_SEED
