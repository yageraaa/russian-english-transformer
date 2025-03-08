import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

if __name__ == "__main__":
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    batch_size = 5
    seq_len = 10
    x = torch.rand(batch_size, seq_len, d_model)

    ff_layer = FeedForwardLayer(d_model, d_ff, dropout)

    output = ff_layer(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output example (first 5 features of the first sequence):", output[0, 0, :5])