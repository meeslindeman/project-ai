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

        for _ in range(num_heads):
            self.W_q = nn.Linear(input_dim, input_dim)
            self.W_k = nn.Linear(input_dim, input_dim)
            self.W_v = nn.Linear(input_dim, input_dim)

        self.W_o = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: (B, N, D) -> (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N) to broadcast over heads and queries
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1) # (B, num_heads, N, N)
        head_outputs = torch.matmul(attn_weights, V)
        
        # Concat heads: (B, num_heads, N, head_dim) -> (B, N, D)
        head_outputs = head_outputs.transpose(1, 2).contiguous().view(B, N, D)

        return self.W_o(head_outputs)

class Classifier(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embed_dim: int, num_classes: int, num_heads: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = Attention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(token_ids)          # [B, N, D]
        attn_output = self.attention(embeds, mask)  # [B, N, D]
        attn_output = F.relu(attn_output)
        pooled = attn_output.mean(dim=1)            # [B, D]
        logits = self.fc(pooled)                    # [B, K]
        return logits