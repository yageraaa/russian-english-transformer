import torch
import torch.nn as nn
from models.modules.layer_norm import LayerNormalization
from models.modules.decoder_layer import DecoderBlock

class Decoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    n_layers = 6
    x = torch.randn(batch_size, seq_len, d_model)
    encoder_output = torch.randn(batch_size, seq_len, d_model)
    src_mask = torch.ones(batch_size, 1, 1, seq_len)
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    decoder = Decoder(d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
    output = decoder(x, encoder_output, src_mask, tgt_mask)

    print(f"Input shape: {x.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Output shape: {output.shape}")