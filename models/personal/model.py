import torch
import torch.nn as nn
import torch.nn.functional as F

from models.personal.lorentz import Lorentz
from models.personal.layer import LorentzMLR, LorentzFC
from models.personal.attention import LorentzAttention

class PersonalMHA(nn.Module):
    def __init__(
        self,
        manifold: Lorentz,
        hidden_dim: int,
        curvature_k: float,
        num_heads: int = 1,
        compute_scores: str = "lorentz_inner",
        concat_operation: str = "direct",
        split_heads: bool = True,
        a_default: float = 0.0,
        attn_debug: bool = False,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.attn = LorentzAttention(
            input_dim=hidden_dim + 1,          # lorentz dim = 1 + spatial dim
            curvature=curvature_k,
            num_heads=num_heads,
            compute_scores=compute_scores,
            concat_operation=concat_operation,
            split_heads=split_heads,
            out_dim=hidden_dim,                # produces output in lorentz dim = 1 + spatial dim
            debug=attn_debug
        )

    def forward(self, x_lorentz: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.attn(x_lorentz, attn_mask=attn_mask)


class PersonalFFN(nn.Module):
    def __init__(self, hidden_dim: int, manifold: Lorentz, reset_params: str = "kaiming", a_default: float = 0.0, activation=nn.Identity()) -> None:
        super().__init__()
        self.lin = LorentzFC(
            in_features=hidden_dim,    # space features only
            out_features=hidden_dim,   
            manifold=manifold,
            reset_params=reset_params,
            a_default=a_default,
            activation=activation,
            do_mlr=False
        )

    def forward(self, x_lorentz: torch.Tensor) -> torch.Tensor:
        # returns a Lorentz vector
        return self.lin(x_lorentz)


class PersonalMLPHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, curvature_k: float, reset_params: str = "kaiming") -> None:
        super().__init__()
        self.mlr = LorentzMLR(
            in_features=hidden_dim,
            out_features=num_classes,
            k=curvature_k,
            reset_params=reset_params,
            activation=nn.Identity(),   
            input_space="lorentz"
        )

    def forward(self, x_lorentz: torch.Tensor) -> torch.Tensor:
        return self.mlr(x_lorentz)


class PersonalModel(nn.Module):
    """
    Convention for personal model:
      - After expmap0, everything is Lorentz vectors of dim (hidden_dim+1).
      - MHA takes Lorentz vectors and returns Lorentz vectors.
      - FFN takes Lorentz vectors and returns Lorentz vectors.
      - Head consumes Lorentz vectors and returns Euclidean logits.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        num_heads: int = 1,
        compute_scores: str = "lorentz_inner",
        concat_operation: str = "direct",
        split_heads: bool = True,
        curvature_k: float = 0.1,
        attn_debug: bool = False,
        attn_mask: torch.Tensor = None,
        a_default: float = 0.0,
        alpha: float = 1.0
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.alpha = alpha

        self.lin_in = nn.Linear(input_dim, hidden_dim)
        self.manifold = Lorentz(curvature_k)

        self.mha_layers = nn.ModuleList([
            PersonalMHA(
                hidden_dim=hidden_dim,
                manifold=self.manifold,
                curvature_k=curvature_k,
                num_heads=num_heads,
                compute_scores=compute_scores,
                concat_operation=concat_operation,
                split_heads=split_heads,
                a_default=a_default,
                attn_debug=attn_debug
            )
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            PersonalFFN(
                hidden_dim=hidden_dim,
                manifold=self.manifold,
                reset_params="kaiming",
                a_default=a_default,
                activation=nn.Identity() 
            )
            for _ in range(num_layers)
        ])

        self.head = PersonalMLPHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            curvature_k=curvature_k,
            reset_params="kaiming"
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
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)                                  
            squeeze_batch = True
        elif x.dim() != 3:               
            raise ValueError(f"x must be [N,D] or [B,N,D], got {x.shape}")

        # euclidean input projection
        x_space = self.lin_in(x)  
        x_space = F.normalize(x_space, p=2, dim=-1) * 0.1

        # map to lorentz manifold
        x_lorentz = self.manifold.expmap0(x_space)  

        for mha, ffn in zip(self.mha_layers, self.ffn_layers):
            y_lorentz = mha(x_lorentz, attn_mask=self.attn_mask)

            x_lorentz = self.manifold.lorentz_residual(x_lorentz, y_lorentz, wx=1.0 - self.alpha, wy=self.alpha)

            y_lorentz = ffn(x_lorentz)

            x_lorentz = self.manifold.lorentz_residual(x_lorentz, y_lorentz, wx=1.0 - self.alpha, wy=self.alpha)

        logits = self.head(x_lorentz)

        if squeeze_batch:
            logits = logits.squeeze(0)
        return logits
