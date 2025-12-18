STACK_SEQ_LEN = 40  # Max cards in dungeon deck (44 total - 4 dealt to room at start)
NUM_CARDS = 45  # Card embeddings: 0=pad, 1-44=actual cards
ACTION_SPACE = 5  # 0-3: Pick Card 1-4, 4: Run/Skip Room

CARD_EMBEDDING_DIM = 32
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2
SCALAR_ENCODER_OUT = 64
HIDDEN_DIM = 128

LR = 8e-3
GAMMA = 0.9
EPS_CLIP = 0.4
K_EPOCHS = 8

TRAIN_MAX_EPISODES = 10000
TRAIN_MAX_STEPS_PER_EPISODE = 200
TRAIN_UPDATE_TIMESTEP = 20
TRAIN_SAVE_INTERVAL = 1000

TRAIN_RESUME_FROM = "/Users/ethan/Desktop/Development/scoundrel/scoundrel/rl/transformer_mcts/checkpoints/ppo_latest.pt"