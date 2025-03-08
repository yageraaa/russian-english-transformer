import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        query = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, attn = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)

if __name__ == "__main__":
    d_model = 512
    h = 8
    dropout = 0.1
    batch_size = 5
    seq_len = 10
    q = torch.rand(batch_size, seq_len, d_model)
    k = torch.rand(batch_size, seq_len, d_model)
    v = torch.rand(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, h, dropout)
    output = mha(q, k, v)

    print("Input shape (q, k, v):", q.shape, k.shape, v.shape)
    print("Output shape:", output.shape)
    print("Output example (first 5 features of the first sequence):", output[0, 0, :5])