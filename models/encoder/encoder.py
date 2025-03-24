import torch
import torch.nn as nn
from models.modules.layer_norm import LayerNormalization
from models.encoder.encoder_layer import EncoderBlock
from models.modules.multihead_attention import MultiHeadAttention
from models.modules.feed_forward import FeedForwardLayer

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

if __name__ == "__main__":
    features = 64
    h = 8
    dropout = 0.1
    self_attn = MultiHeadAttention(features, h, dropout)
    feed_forward = FeedForwardLayer(features, features * 4, dropout)
    encoder_blocks = nn.ModuleList([EncoderBlock(features, h, features * 4, dropout)])
    encoder = Encoder(features, encoder_blocks)
    batch_size = 5
    seq_len = 10
    x = torch.rand(batch_size, seq_len, features)
    mask = torch.ones(batch_size, seq_len, seq_len)
    output = encoder(x, mask)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output example (first 5 features of the first sequence):", output[0, 0, :5])
