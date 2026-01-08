"""
Configuration constants for AlphaGo-style MCTS.
"""

# Model paths (relative to alpha_scoundrel/)
POLICY_LARGE_CHECKPOINT = "policy/policy_large/checkpoints/run_20260107_132912/best_model.pt"
POLICY_SMALL_CHECKPOINT = "policy/policy_small/checkpoints/run_20260107_173211/policy_small_epoch_50.pt"
VALUE_LARGE_CHECKPOINT = "value/value_large/checkpoints/run_20260108_091813/best_model.pt"

# MCTS parameters (fewer simulations needed with neural guidance)
ALPHAGO_NUM_SIMULATIONS = 100
ALPHAGO_C_PUCT = 2  # PUCT exploration constant
ALPHAGO_VALUE_WEIGHT = 0  # λ in [0,1]: 0=pure value net, 1=pure rollout
ALPHAGO_MAX_DEPTH = 120  # Max rollout depth

# Exploration Noise (AlphaGo Zero style)
DIRICHLET_ALPHA = 0.5   # Concentration parameter (0.3 for chess, 0.03 for Go, ~1.0 for Scoundrel/small action space)
DIRICHLET_EPSILON = 0.25 # Noise weight (0.25 is standard)

# Parallelization
# NOTE: Workers use CPU to avoid GPU/MPS contention issues
# Set to 0 or 1 to disable parallelization (use sequential search)
# Higher values provide speedup but require more CPU cores
ALPHAGO_NUM_WORKERS = 1  # Root parallelization workers
ALPHAGO_TRANSPOSITION_TABLE_SIZE = 100000

# Evaluation
EVAL_NUM_GAMES = 10
EVAL_SEED = 42

