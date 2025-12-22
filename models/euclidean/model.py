import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.euclidean.attenttion import Attention

class GraphClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 1,
        num_layers: int = 2,
        attn_mask: torch.Tensor | None = None,
        alpha: float = 0.5
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.alpha = alpha

        self.lin_in = nn.Linear(in_dim, hidden_dim)

        self.attn_layers = nn.ModuleList([
            Attention(input_dim=hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin_in(x)                                          # (N, hidden)
        x = F.normalize(x, p=2, dim=-1) * 1.0      

        x = x.unsqueeze(0)                                         # (1, N, hidden)

        for attn in self.attn_layers:
            y = attn(x, adj_mask=self.attn_mask)                   # (1, N, hidden)
            y = F.relu(y)
            x = (1.0 - self.alpha) * x + self.alpha * y

        x = x.squeeze(0)                                           # (N, hidden)
        out = self.fc(x)                                           # (N, C)
        return out