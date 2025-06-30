import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * (x / rms)


if __name__ == "__main__":
    features = 64
    batch_size = 10
    seq_len = 20
    x = torch.rand(batch_size, seq_len, features)
    rms_norm = RMSNorm(features)
    output = rms_norm(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output example (first 5 features of the first sequence):", output[0, 0, :5])
