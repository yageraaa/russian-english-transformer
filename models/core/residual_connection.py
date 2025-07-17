import torch
import torch.nn as nn
from models.core.layer_norm import LayerNormalization
from models.core.rms_norm import RMSNorm


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float, norm_type: str = "rms"):
        super().__init__()
        if norm_type == "layer":
            self.norm = LayerNormalization(features)
        elif norm_type == "rms":
            self.norm = RMSNorm(features)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Choose 'layer' or 'rms'.")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


if __name__ == "__main__":
    features = 64
    dropout = 0.1
    batch_size = 5
    seq_len = 10
    x = torch.rand(batch_size, seq_len, features)
    sublayer = nn.Linear(features, features)
    res_connection_layer = ResidualConnection(features, dropout, norm_type="layer")
    output_layer = res_connection_layer(x, sublayer)
    res_connection_rms = ResidualConnection(features, dropout, norm_type="rms")
    output_rms = res_connection_rms(x, sublayer)

    print("Input shape:", x.shape)
    print("Output shape (LayerNorm):", output_layer.shape)
    print("Output example (LayerNorm, first 5 features of the first sequence):", output_layer[0, 0, :5])
    print("Output shape (RMSNorm):", output_rms.shape)
    print("Output example (RMSNorm, first 5 features of the first sequence):", output_rms[0, 0, :5])