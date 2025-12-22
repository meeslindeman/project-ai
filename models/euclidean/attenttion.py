import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        assert input_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim, bias=False)
        self.W_k = nn.Linear(input_dim, input_dim, bias=False)
        self.W_v = nn.Linear(input_dim, input_dim, bias=False)
        self.W_o = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # (B, N, D) -> (B, H, N, Hd)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, N, N)

        if adj_mask is not None:
            if adj_mask.dim() == 2:          # (N, N)
                m = adj_mask[None, None, :, :]        # (1, 1, N, N)
            elif adj_mask.dim() == 3:        # (B, N, N)
                m = adj_mask[:, None, :, :]           # (B, 1, N, N)
            else:
                raise ValueError(f"adj_mask must be (N,N) or (B,N,N), got {adj_mask.shape}")

            scores = scores.masked_fill(~m, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, N, Hd)

        out = out.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        
        return self.W_o(out)
