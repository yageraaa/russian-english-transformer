from imports import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)