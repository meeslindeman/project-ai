import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hypformer.lorentz import Lorentz
from models.hypformer.layer import HypLinear
from models.hypformer.attention import HypformerAttention
from models.hypformer.decoder import HyperbolicCLS, HyperbolicLinear

class LorentzMHA(nn.Module):
    def __init__(self, manifold: Lorentz, hidden_dim: int, num_heads: int = 1, att_type: str = 'full') -> None:
        super().__init__()
        self.attention = HypformerAttention(
            manifold=manifold,
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            att_type=att_type,
            num_heads=num_heads,
            use_weight=True,
            heads_concat=False
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # input to attention is expected to be in Euclidean space
        x_lorentz = self.attention(x, attn_mask=attn_mask)
        return x_lorentz


class LorentzFFN(nn.Module):
    def __init__(self, manifold: Lorentz, hidden_dim: int, bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.manifold = manifold

        # HypLinear adds time dimension internally
        self.lin = HypLinear(
            manifold=manifold,
            in_features=hidden_dim - 1, 
            out_features=hidden_dim - 1,
            bias=bias
        )

    def forward(self, x_lorentz: torch.Tensor) -> torch.Tensor:
        # x_lorentz is on Lorentz manifold so let HypLinear handle it directly
        return self.lin(x_lorentz)


class LorentzMLPHead(nn.Module):
    def __init__(self, manifold: Lorentz, hidden_dim: int, num_classes: int, decoder_type: str = "linear", bias: bool = True) -> None:
        super().__init__()
        dim_lorentz = hidden_dim
        
        if decoder_type == "cls":
            self.decoder = HyperbolicCLS(
                manifold=manifold,
                input_dim=dim_lorentz,
                num_classes=num_classes,
                bias=bias,
            )
        elif decoder_type == "linear":
            self.decoder = HyperbolicLinear(
                manifold=manifold,
                input_dim=dim_lorentz,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(self, x_lorentz: torch.Tensor) -> torch.Tensor:
        return self.decoder(x_lorentz)


class Hypformer(nn.Module):
    """
    Convention (driven by the HypformerAttention/HypLinear implementation):
      - MHA consumes Euclidean spatial features (hidden_dim)
      - MHA outputs Lorentz vectors (hidden_dim+1)
      - Next layer's MHA must receive spatial part only: x_lor[..., 1:]
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_classes: int, 
        num_layers: int = 1,
        num_heads: int = 1,
        att_type: str = "full", 
        decoder_type: str = "linear", 
        k: float = 1.0,
        attn_mask: torch.Tensor = None,
        alpha: float = 1.0,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.manifold = Lorentz(k)

        self.mha_layers = nn.ModuleList([
            LorentzMHA(self.manifold, hidden_dim=hidden_dim, num_heads=num_heads, att_type=att_type)
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            LorentzFFN(self.manifold, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        self.head = LorentzMLPHead(
            manifold=self.manifold,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            decoder_type=decoder_type,
        )
    
    @staticmethod
    def _minkowski_norm_sq(x: torch.Tensor) -> torch.Tensor:
        time = x[..., :1]   
        space = x[..., 1:] 
        norm_sq = -time * time + torch.sum(space * space, dim=-1, keepdim=True)
        return norm_sq

    def _is_on_manifold(self, x: torch.Tensor, manifold: Lorentz, tol: float = 1e-4, log_details: bool = False) -> bool:
        # k_val = manifold.k().item()
        k_val = 1.0
        target = -1.0 / k_val

        norm_sq = self._minkowski_norm_sq(x)
        diff = norm_sq - target
        max_diff = diff.abs().max().item()

        print(max_diff < tol)

        if log_details:
            print("[manifold check] k=%.6f", k_val)
            print("  target Minkowski norm = %.6f", target)
            print("  max |<x,x>_L - target| = %.6e", max_diff)

        return None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # hypformer expects unbatched input [N, D]
        if x.dim() != 2:
            raise ValueError(f"Expected x to be [N, D], got {x.shape}")

        # euclidean input projection and dropout
        x_space = self.lin_in(x) 
        x_space = self.dropout(x_space)
        x_space = F.normalize(x_space, p=2, dim=-1) * 1.0

        # map to lorentz manifold
        x_lorentz = self.manifold.expmap0(x_space)

        for mha, ffn in zip(self.mha_layers, self.ffn_layers):
            y_lorentz = mha(x_lorentz, attn_mask=self.attn_mask)

            # residual connection
            x_lorentz = self.manifold.mid_point(
                torch.stack([x_lorentz, y_lorentz], dim=1)
            )

            z_lorentz = ffn(x_lorentz) 

            # residual connection
            x_lorentz = self.manifold.mid_point(
                torch.stack((x_lorentz, z_lorentz), dim=1)
            )

        logits = self.head(x_lorentz)

        return logits
        

