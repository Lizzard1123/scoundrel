"""
Configuration constants for AlphaGo-style MCTS.
"""

# Model paths (relative to alpha_scoundrel/)
POLICY_LARGE_CHECKPOINT = "policy/policy_large/checkpoints/run_20251229_194640/policy_large_epoch_40.pt"
POLICY_SMALL_CHECKPOINT = "policy/policy_small/checkpoints/run_20251229_184938/policy_small_epoch_10.pt"
VALUE_LARGE_CHECKPOINT = "value/value_large/checkpoints/100e_193mse/value_large_epoch_100.pt"

# MCTS parameters (fewer simulations needed with neural guidance)
ALPHAGO_NUM_SIMULATIONS = 3000
ALPHAGO_C_PUCT = 1.0  # PUCT exploration constant
ALPHAGO_VALUE_WEIGHT = 0.5  # λ in [0,1]: 0=pure value net, 1=pure rollout
ALPHAGO_MAX_DEPTH = 120  # Max rollout depth

# Parallelization
ALPHAGO_NUM_WORKERS = 8  # Root parallelization workers
ALPHAGO_TRANSPOSITION_TABLE_SIZE = 100000

# Evaluation
EVAL_NUM_GAMES = 10
EVAL_SEED = 42

