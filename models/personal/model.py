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

    @staticmethod
    def _minkowski_norm_sq(x: torch.Tensor) -> torch.Tensor:
        time = x[..., :1]   
        space = x[..., 1:] 
        norm_sq = -time * time + torch.sum(space * space, dim=-1, keepdim=True)
        return norm_sq

    def _is_on_manifold(self, x: torch.Tensor, manifold: Lorentz, tol: float = 1e-4, log_details: bool = False) -> bool:
        k_val = manifold.k().item()
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
        x = self.lin_in(x)  # [N, D]
        x = F.normalize(x, p=2, dim=-1) * 1.0
        
        # attention expects input in Lorentz model
        x = self.manifold.expmap0(x).unsqueeze(0)  # [1, N, 1+D]

        alpha = 0.5
        for attn in self.attn_layers:
            y = attn(x, attn_mask=self.attn_mask)
            x = self.manifold.lorentz_residual(x, y, wx=1 - alpha, wy=alpha)

        x = x.squeeze(0)    # [N, 1+D]
        out = self.fc(x)    # [N, C]
        return out
