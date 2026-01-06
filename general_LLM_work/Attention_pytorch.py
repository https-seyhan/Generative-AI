import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_k)
        self.scale = d_k ** 0.5

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        Q = self.W_Q(x)  # (batch, seq_len, d_k)
        K = self.W_K(x)  # (batch, seq_len, d_k)
        V = self.W_V(x)  # (batch, seq_len, d_k)

        # Scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)

        # Output = weights * V
        output = torch.matmul(weights, V)
        return output, weights


# Example usage
batch, seq_len, d_model, d_k = 2, 5, 64, 32
x = torch.randn(batch, seq_len, d_model)

attention = ScaledDotProductAttention(d_model, d_k)
out, attn_weights = attention(x)

print("Output shape:", out.shape)
print("Attention weights shape:", attn_weights.shape)
