import torch
import torch.nn as nn
from models.core.rms_norm import RMSNorm
from models.core.swiglu import SwiGLU
from models.core.qk_norm import QKNorm
from models.core.positional_encoding import PositionalEncoding
from models.core.embeddings import InputEmbeddings
from models.core.residual_connection_rms import ResidualConnectionWithRMSNorm


class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int = 512,
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1,
                 d_ff: int = 2048):
        super().__init__()
        self.embed = InputEmbeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, seq_len)
        self.layers = nn.ModuleList([
            SelfAttentionDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.projection = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(self.embed(x))
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.projection(x)

    def generate(self, start_tokens: torch.Tensor, max_len: int,
                 end_token_id: int) -> torch.Tensor:
        batch_size, device = start_tokens.size(0), start_tokens.device
        tokens = start_tokens

        for _ in range(max_len):
            mask = torch.tril(torch.ones(tokens.size(1), tokens.size(1))).to(device)
            mask = mask.unsqueeze(0).unsqueeze(0)
            logits = self(tokens, mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

            if (next_token == end_token_id).all():
                break

        return tokens


class SelfAttentionDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = QKNorm(d_model, num_heads, dropout)
        self.cross_attention = QKNorm(d_model, num_heads, dropout)
        self.feed_forward = SwiGLU(d_model, d_ff, dropout)
        self.residuals = nn.ModuleList([
            ResidualConnectionWithRMSNorm(d_model, dropout) for _ in range(3)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.residuals[0](x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residuals[1](x, lambda x: torch.zeros_like(x))
        x = self.residuals[2](x, self.feed_forward)
        return x


if __name__ == "__main__":
    vocab_size = 10000
    seq_len = 128
    batch_size = 16

    model = DecoderOnlyModel(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        d_ff=2048
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    output = model(x, mask)

    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

