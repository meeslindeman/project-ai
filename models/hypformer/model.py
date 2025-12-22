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

        # only one linear layer in the feedforward network as hyperbolic layers are inherently non-linear
        self.lin = HypLinear(
            manifold=manifold,
            in_features=hidden_dim, # HypLinear adds time dimension internally
            out_features=hidden_dim,
            bias=bias,
            dropout=dropout,
            manifold_out=manifold
        )

    def forward(self, x_lorentz: torch.Tensor) -> torch.Tensor:
        # x_lorentz is on Lorentz manifold so let HypLinear handle it directly
        return self.lin(x_lorentz, x_manifold="hyp")


class LorentzMLPHead(nn.Module):
    def __init__(self, manifold: Lorentz, hidden_dim: int, num_classes: int, decoder_type: str = "linear", bias: bool = True) -> None:
        super().__init__()
        dim_lorentz = hidden_dim + 1
        
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

        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.manifold = Lorentz(k)

        self.mha_layers = nn.ModuleList([
            LorentzMHA(self.manifold, hidden_dim=hidden_dim, num_heads=num_heads, att_type=att_type)
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            LorentzFFN(self.manifold, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.head = LorentzMLPHead(
            manifold=self.manifold,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            decoder_type=decoder_type,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # hypformer expects unbatched input [N, D]
        if x.dim() != 2:
            raise ValueError(f"Expected x to be [N, D], got {x.shape}")

        # euclidean input projection
        x_space = self.lin_in(x) 
        x_space = F.normalize(x_space, p=2, dim=-1) * 0.1

        x_lorentz = None
        for mha, ffn in zip(self.mha_layers, self.ffn_layers):
            # mha consumes euclidean coords and outputs lorentz vectors
            y_lorentz = mha(x_space, attn_mask=self.attn_mask)

            # residual connection
            if x_lorentz is None:
                x_lorentz = y_lorentz
            else:
                # non-manifold-correct residual (ambient-space hack)
                x_lorentz = (1.0 - self.alpha) * x_lorentz + self.alpha * y_lorentz

            # feedforward network operates in lorentz space
            y_lorentz = ffn(x_lorentz) 

            # residual connection
            x_lorentz = (1.0 - self.alpha) * x_lorentz + self.alpha * y_lorentz

            # extract space part for next MHA layer as it expects euclidean input
            x_space = x_lorentz[..., 1:]

        # decode from lorentz space to logits
        logits = self.head(x_lorentz)

        return logits
        

