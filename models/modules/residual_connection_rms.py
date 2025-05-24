import torch
import torch.nn as nn
from models.modules.rms_norm import RMSNorm

class ResidualConnectionWithRMSNorm(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = RMSNorm(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

if __name__ == "__main__":
    features = 64
    dropout = 0.1
    batch_size = 5
    seq_len = 10
    x = torch.rand(batch_size, seq_len, features)
    sublayer = nn.Linear(features, features)
    res_connection = ResidualConnectionWithRMSNorm(features, dropout)
    output = res_connection(x, sublayer)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output example (first 5 features of the first sequence):", output[0, 0, :5])