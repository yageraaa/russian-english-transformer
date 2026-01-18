import torch
import torch.nn as nn
from models.core.rms_norm import RMSNorm
from models.transformer.decoder_layer import DecoderBlockWithNewTechniques


class DecoderWithNewTechniques(nn.Module):
    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlockWithNewTechniques(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


if __name__ == "__main__":
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    dropout = 0.1

    decoder = DecoderWithNewTechniques(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    )
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model)
    encoder_output = torch.randn(batch_size, seq_len, d_model)
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    output = decoder(x, encoder_output, src_mask, tgt_mask)

    print(f"Input shape: {x.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output shape: {output.shape}")