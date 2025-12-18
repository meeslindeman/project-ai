import torch
import torch.nn as nn
import torch.nn.functional as F

from models.personal.layer import LorentzMLR
from models.personal.lorentz import Lorentz
from models.personal.attention import LorentzAttention

class GraphClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        curvature_k: float = 0.1,
        num_heads: int = 1,
        compute_scores: str = "lorentz_inner",
        concat_operation: str = "direct",
        split_heads: bool = True,
        a_default: float = 0.0,
        attn_debug: bool = False,
        num_layers: int = 2,
        attn_mask: torch.Tensor = None
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.manifold = Lorentz(curvature_k)

        self.attn_layers = nn.ModuleList([
            LorentzAttention(
                input_dim=hidden_dim + 1,
                curvature=curvature_k,
                num_heads=num_heads,
                compute_scores=compute_scores,
                concat_operation=concat_operation,
                split_heads=split_heads,
                out_dim=hidden_dim,
                debug=attn_debug,
            )
            for _ in range(num_layers)
        ])

        self.fc = LorentzMLR(
            in_features=hidden_dim,
            out_features=num_classes,
            k=curvature_k,
            reset_params="kaiming",
            input="lorentz",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin_in(x)  # [N, D]
        x = F.normalize(x, p=2, dim=-1) * 1.0

        # attention expects input in Lorentz model
        x = self.manifold.expmap0(x).unsqueeze(0)  # [1, N, 1+D]

        for attn in self.attn_layers:
            x = attn(x, attn_mask=self.attn_mask)

        x = x.squeeze(0)         # [N, 1+D]
        return self.fc(x)        # [N, C]
