import torch
import torch.nn as nn
from models.modules.multihead_attention import MultiHeadAttention
from models.modules.feed_forward import FeedForwardLayer
from models.modules.residual_connection import ResidualConnection

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardLayer(d_model, d_ff, dropout)
        self.residuals = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residuals[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residuals[2](x, self.feed_forward)
        return x

if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 256
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    x = torch.randn(batch_size, seq_len, d_model)
    encoder_output = torch.randn(batch_size, seq_len, d_model)
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    decoder_block = DecoderBlock(d_model, n_heads, d_ff, dropout)
    output = decoder_block(x, encoder_output, src_mask, tgt_mask)

    print(f"Input shape: {x.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output shape: {output.shape}")