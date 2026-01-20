import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.personal.layer import LorentzFC
from models.personal.lorentz import Lorentz

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

        self.manifold = Lorentz(1.0)

        if lorentz_map:
            self.lorentz_fc = LorentzFC(
                in_features=hidden_dim + 1, # lorentz dim
                out_features=num_classes + 1, # lorentz dim
                manifold=self.manifold,
                reset_params="kaiming",
                a_default=0.0,
                do_mlr=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lorentz_map:
            x = self.manifold.expmap0(x)
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
        dropout: float = 0.0,
        lorentz_map: bool = False,
        use_ffn: bool = False
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.use_ffn = use_ffn
        self.lorentz_map = lorentz_map

        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.mha_layers = nn.ModuleList([MHA(input_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([FFN(input_dim=hidden_dim, hidden_dim=hidden_dim) for _ in range(num_layers)])

        self.head = MLPHead(hidden_dim=hidden_dim, num_classes=num_classes, lorentz_map=lorentz_map)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)                                  
            squeeze_batch = True
        elif x.dim() != 3:               
            raise ValueError(f"x must be [N,D] or [B,N,D], got {x.shape}")
        
        # project to hidden dim and dropout
        x = self.lin_in(x)                      
        x = self.dropout(x)
        x = F.normalize(x, p=2, dim=-1) * 1.0                                                 

        for mha, ffn in zip(self.mha_layers, self.ffn_layers):
            x = mha(x, adj_mask=self.attn_mask)                   

            if self.use_ffn:
                x = ffn(x)

        # per node logits
        logits = self.head(x)                   # [B, N, C]   

        if squeeze_batch:
            logits = logits.squeeze(0)

        return logits