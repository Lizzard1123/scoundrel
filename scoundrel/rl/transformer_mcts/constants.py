# Environment
STACK_SEQ_LEN = 40  # Max cards in face down dungeon deck
NUM_CARDS = 53
ACTION_SPACE = 5  # 0-3: Pick Card 1-4, 4: Run/Skip Room

# Architecture
CARD_EMBEDDING_DIM = 32
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2
SCALAR_ENCODER_OUT = 64
HIDDEN_DIM = 128

# PPO learning
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4