import torch
import torch.nn as nn
from models.modules.layer_norm import LayerNormalization
from models.modules.encoder_layer import EncoderBlock

class Encoder(nn.Module):
    def __init__(self,d_model: int,num_layers: int,num_heads: int,d_ff: int,dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout)for _ in range(num_layers)])
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

if __name__ == "__main__":
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    encoder = Encoder(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=dropout
    )
    batch_size = 5
    seq_len = 10
    x = torch.rand(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len, seq_len)
    output = encoder(x, mask)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output example (first 5 features):", output[0, 0, :5])