import torch
import torch.nn as nn
from models.modules.multihead_attention import MultiHeadAttention
from models.modules.residual_connection import ResidualConnection
from models.modules.feed_forward import FeedForwardLayer


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attn: MultiHeadAttention, feed_forward: FeedForwardLayer, dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.residuals = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.residuals[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.residuals[1](x, self.feed_forward)

if __name__ == "__main__":
    features = 64
    h = 8
    dropout = 0.1
    self_attn = MultiHeadAttention(features, h, dropout)
    feed_forward = FeedForwardLayer(features, features * 4, dropout)
    encoder_block = EncoderBlock(features, self_attn, feed_forward, dropout)
    batch_size = 5
    seq_len = 10
    x = torch.rand(batch_size, seq_len, features)
    mask = torch.ones(batch_size, seq_len, seq_len)
    output = encoder_block(x, mask)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output example (first 5 features of the first sequence):", output[0, 0, :5])