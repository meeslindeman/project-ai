import torch
import torch.nn as nn
import torch.nn.functional as F

from models.personal.layer import LorentzMLR
from models.personal.lorentz import Lorentz
from models.personal.attention import LorentzAttention

class Classifer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        curvature_k: float = 0.1,
        num_heads: int = 1,
        compute_scores: str = "lorentz_inner",
        value_agg: str = "midpoint",
        concat_operation: str = "direct",
        a_default: float = 0.0,
        split_qkv: bool = False,
        attn_debug: bool = False,
        num_layers: int = 2,
        attn_mask: torch.Tensor = None
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.curvature_k = curvature_k
        self.num_layers = num_layers
        self.attn_mask = attn_mask

        self.lin_in = nn.Linear(in_dim, hidden_dim)

        self.manifold = Lorentz(curvature_k)

        self.attn_layers = nn.ModuleList(
            [
                LorentzAttention(
                    input_dim=hidden_dim + 1,
                    curvature=curvature_k,
                    num_heads=num_heads,
                    compute_scores=compute_scores,
                    value_agg=value_agg,
                    concat_operation=concat_operation,
                    out_dim=hidden_dim,
                    a_default=a_default,
                    split_qkv=split_qkv,
                    debug=attn_debug,
                    attn_mask=attn_mask
                )
                for _ in range(num_layers)
            ]
        )

        self.fc = LorentzMLR(
            in_features=hidden_dim,
            out_features=num_classes,
            k=curvature_k,
            reset_params="kaiming",
            input="lorentz",
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> torch.Tensor:
        x = self.lin_in(x)  # [N, D]

        x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x = x * 1.0

        x = self.manifold.safe_expmap0(x).unsqueeze(0)  # [1, N, 1+D]

        for attn in self.attn_layers:
            x_in = x
            x = attn(x, attn_mask=self.attn_mask)
            x = 0.5 * x + 0.5 * x_in
            x = self.manifold.proj(x)

        x = x.squeeze(0)         # [N, 1+D]
        return self.fc(x)        # [N, C]
