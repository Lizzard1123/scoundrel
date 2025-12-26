STACK_SEQ_LEN = 40
NUM_CARDS = 45
ACTION_SPACE = 5

# MLP architecture constants
SCALAR_ENCODER_OUT = 64
HIDDEN_DIM = 32

LR = 1e-3
BATCH_SIZE = 256
EPOCHS = 1000
TRAIN_VAL_SPLIT = 0.9

DEFAULT_MCTS_LOGS_DIR = "scoundrel/rl/mcts/logs/collected_games"
MAX_GAMES = None

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "policy_small_epoch_"
CHECKPOINT_INTERVAL = 10

