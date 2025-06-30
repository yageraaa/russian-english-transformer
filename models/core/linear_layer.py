import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)


if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    vocab_size = 256
    projection_layer = ProjectionLayer(d_model, vocab_size)
    decoder_output = torch.randn(batch_size, seq_len, d_model)
    logits = projection_layer(decoder_output)

    print(f"Input shape: {decoder_output.shape}")
    print(f"Output shape: {logits.shape}")
