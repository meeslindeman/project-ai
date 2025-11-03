import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

class Classifier(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embed_dim: int, num_classes: int, **kwargs) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = Attention(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(token_ids)          # [B, N, D]
        attn_output = self.attention(embeds, mask)  # [B, N, D]
        pooled = attn_output.mean(dim=1)            # [B, D]
        logits = self.fc(pooled)                    # [B, K]
        return logits