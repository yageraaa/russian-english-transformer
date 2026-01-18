import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.w3 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish = self.w1(x) * torch.sigmoid(self.w1(x))
        gate = self.w2(x)
        x = swish * gate
        x = self.dropout(x)
        return self.w3(x)


if __name__ == "__main__":
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    batch_size = 5
    seq_len = 10
    x = torch.rand(batch_size, seq_len, d_model)
    swiglu_layer = SwiGLU(d_model, d_ff, dropout)
    output = swiglu_layer(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output example (first 5 features of the first sequence):", output[0, 0, :5])