import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.personal.layer import LorentzMLR

class MHA(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 1, bias: bool = False) -> None:
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.W_q = nn.Linear(input_dim, input_dim, bias=bias)
        self.W_k = nn.Linear(input_dim, input_dim, bias=bias)
        self.W_v = nn.Linear(input_dim, input_dim, bias=bias)
        self.W_o = nn.Linear(input_dim, input_dim, bias=bias)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape

        Q = self.W_q(x) 
        K = self.W_k(x)
        V = self.W_v(x)

        # [B, N, D] -> [B, H, N, Hd]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, Hd]

        # allow batch or single mask
        if adj_mask is not None:
            if adj_mask.dim() == 2:                 # [N, N]
                m = adj_mask[None, None, :, :]      # [1, 1, N, N]
            elif adj_mask.dim() == 3:               # [B, N, N]
                m = adj_mask[:, None, :, :]         # [B, 1, N, N]
            else:
                raise ValueError(f"adj_mask must be [N,N] or [B,N,N], got {adj_mask.shape}")

            scores = scores.masked_fill(~m, float("-inf"))

        attn = F.softmax(scores, dim=-1)    # [B, H, N, N]
        out = torch.matmul(attn, V)         # [B, H, N, Hd]

        # [B, H, N, Hd] -> [B, N, D]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        
        return self.W_o(out)


class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.lin2 = nn.Linear(hidden_dim, input_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


class MLPHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, bias: bool = False, lorentz_map: bool = False) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden_dim, num_classes, bias=bias)
        self.lorentz_map = lorentz_map
        
        if lorentz_map:
            self.lorentz_fc = LorentzMLR(hidden_dim, num_classes, input_space="euclidean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lorentz_map:
            return self.lorentz_fc(x)
        return self.lin(x)


class EuclideanModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 1,
        num_layers: int = 1,
        attn_mask: torch.Tensor | None = None,
        alpha: float = 1.0,
        lorentz_map: bool = False
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.alpha = alpha
        self.lorentz_map = lorentz_map

        self.lin_in = nn.Linear(input_dim, hidden_dim)

        self.mha_layers = nn.ModuleList([MHA(input_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([FFN(input_dim=hidden_dim, hidden_dim=hidden_dim) for _ in range(num_layers)])

        self.head = MLPHead(hidden_dim=hidden_dim, num_classes=num_classes, lorentz_map=lorentz_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)                                  
            squeeze_batch = True
        
        # project to hidden dim and normalize
        x = self.lin_in(x)                      # [B, N, D]             
        x = F.normalize(x, p=2, dim=-1) * 1.0                                                    

        for mha, ffn in zip(self.mha_layers, self.ffn_layers):
            y = mha(x, adj_mask=self.attn_mask)                   

            # residual connection
            x = (1.0 - self.alpha) * x + self.alpha * y

            y = ffn(x)

            # residual connection
            x = (1.0 - self.alpha) * x + self.alpha * y

        # per node logits
        logits = self.head(x)                   # [B, N, C]   

        if squeeze_batch:
            logits = logits.squeeze(0)

        return logits