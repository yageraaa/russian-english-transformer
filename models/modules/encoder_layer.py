import torch
import torch.nn as nn
from models.modules.multihead_attention import MultiHeadAttention
from models.modules.residual_connection import ResidualConnection
from models.modules.feed_forward import FeedForwardLayer

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardLayer(d_model, d_ff, dropout)
        self.residuals = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.residuals[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.residuals[1](x, self.feed_forward)

if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    encoder_block = EncoderBlock(d_model, num_heads, d_ff, dropout)
    batch_size = 5
    seq_len = 10
    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len, seq_len)
    output = encoder_block(x, mask)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output example (first 5 features):", output[0, 0, :5])