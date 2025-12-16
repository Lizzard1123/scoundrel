import torch
import torch.nn as nn
import torch.nn.functional as F
from scoundrel.rl.transformer_mlp.constants import ACTION_SPACE, CARD_EMBEDDING_DIM, HIDDEN_DIM, NUM_CARDS, SCALAR_ENCODER_OUT, TRANSFORMER_NHEAD, TRANSFORMER_NLAYERS

class ScoundrelNet(nn.Module):
    """
    Hybrid Architecture:
    - Transformer for the 'Known Stack' (Sequence planning)
    - MLP for Current Room & Status (Immediate tactics)
    """
    def __init__(self, scalar_input_dim):
        super(ScoundrelNet, self).__init__()

        # -- 1. Sequence Encoder (The "Planner") --
        self.embedding = nn.Embedding(NUM_CARDS, CARD_EMBEDDING_DIM, padding_idx=0)
        # Using a small Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=CARD_EMBEDDING_DIM, nhead=TRANSFORMER_NHEAD, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_NLAYERS)

        # -- 2. Scalar Encoder (The "Tactician") --
        self.scalar_fc = nn.Linear(scalar_input_dim, SCALAR_ENCODER_OUT)

        # -- 3. Fusion & Heads --
        # Combined dim: Transformer output (flattened or pooled) + Scalar output
        combined_dim = CARD_EMBEDDING_DIM + SCALAR_ENCODER_OUT

        self.shared_layer = nn.Linear(combined_dim, HIDDEN_DIM)

        # Actor Head (Policy)
        self.actor = nn.Linear(HIDDEN_DIM, ACTION_SPACE)
        # Critic Head (Value)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, scalar_data, sequence_data):
        # Process Known Stack
        # sequence_data shape: [Batch, 52]
        embedded = self.embedding(sequence_data) # [Batch, 52, 32]

        # We only care about the first few "next" cards and general composition,
        trans_out = self.transformer(embedded)
        seq_features = torch.mean(trans_out, dim=1) # Average pooling [Batch, 32]

        # Process Scalars
        scal_features = F.relu(self.scalar_fc(scalar_data)) # [Batch, 64]

        # Fusion
        combined = torch.cat((seq_features, scal_features), dim=1)
        x = F.relu(self.shared_layer(combined))

        return self.actor(x), self.critic(x)
