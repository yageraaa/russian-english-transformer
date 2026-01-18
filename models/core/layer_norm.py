import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

if __name__ == "__main__":
    features = 64
    batch_size = 10
    seq_len = 20
    x = torch.rand(batch_size, seq_len, features)
    layer_norm = LayerNormalization(features)
    output = layer_norm(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output example (first 5 features of the first sequence):", output[0, 0, :5])