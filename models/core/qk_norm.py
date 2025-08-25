import torch
import torch.nn as nn
import math


class QKNorm(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        q_norm = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-6)
        k_norm = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
        scores = torch.matmul(q_norm, k_norm.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(output)


if __name__ == "__main__":
    d_model = 512
    n_heads = 8
    dropout = 0.1
    batch_size = 5
    seq_len = 10
    q = torch.rand(batch_size, seq_len, d_model)
    k = torch.rand(batch_size, seq_len, d_model)
    v = torch.rand(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, 5:] = 0
    qk_norm = QKNorm(d_model, n_heads, dropout)
    output = qk_norm(q, k, v, mask)

    print("Input shape (q, k, v):", q.shape, k.shape, v.shape)
    print("Output shape:", output.shape)