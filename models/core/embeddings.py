import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.scale = d_model ** 0.5

    def forward(self, x):
        return self.embedding(x) * self.scale

if __name__ == "__main__":
    vocab_size = 256
    d_model = 512
    embeddings = InputEmbeddings(d_model, vocab_size)
    batch_size = 5
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = embeddings(input_ids)

    print("Input IDs shape:", input_ids.shape)
    print("Output embeddings shape:", output.shape)
    print("Output example:", output[0, :3, :5])