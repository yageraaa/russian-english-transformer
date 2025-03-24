import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)

        self.register_buffer('encoding', encoding)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

if __name__ == "__main__":
    d_model = 512
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    batch_size = 5
    seq_len = 10
    embeddings = torch.rand(batch_size, seq_len, d_model)
    output = pe(embeddings)

    print("Input embeddings shape:", embeddings.shape)
    print("Output with positional encoding shape:", output.shape)
    print("Output example:", output[0, :3, :5])