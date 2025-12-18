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

        self.embedding = nn.Embedding(NUM_CARDS, CARD_EMBEDDING_DIM, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=CARD_EMBEDDING_DIM, nhead=TRANSFORMER_NHEAD, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_NLAYERS)

        self.scalar_fc = nn.Linear(scalar_input_dim, SCALAR_ENCODER_OUT)

        combined_dim = CARD_EMBEDDING_DIM + SCALAR_ENCODER_OUT

        self.shared_layer = nn.Linear(combined_dim, HIDDEN_DIM)

        self.actor = nn.Linear(HIDDEN_DIM, ACTION_SPACE)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, scalar_data, sequence_data):
        embedded = self.embedding(sequence_data)

        trans_out = self.transformer(embedded)
        seq_features = torch.mean(trans_out, dim=1)

        scal_features = F.relu(self.scalar_fc(scalar_data))

        combined = torch.cat((seq_features, scal_features), dim=1)
        x = F.relu(self.shared_layer(combined))

        return self.actor(x), self.critic(x)
