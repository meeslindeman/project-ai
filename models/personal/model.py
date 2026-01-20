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
        curvature: float,
        num_heads: int = 1,
        compute_scores: str = "lorentz_inner",
        head_fusion: str = "midpoint",
        split_heads: bool | None = True,
        a_default: float = 0.0,
        reset_params: str = "lorentz_kaiming"
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.attn = LorentzAttention(
            input_dim=hidden_dim + 1,          
            curvature=curvature,
            num_heads=num_heads,
            compute_scores=compute_scores,
            head_fusion=head_fusion,
            split_heads=split_heads,
            out_dim=hidden_dim,               
            reset_params=reset_params,
            a_default=a_default
        )

    def forward(self, x_lorentz: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.attn(x_lorentz, attn_mask=attn_mask)


class PersonalFFN(nn.Module):
    def __init__(self, hidden_dim: int, manifold: Lorentz, curvature: float, reset_params: str = "lorentz_kaiming", a_default: float = 0.0) -> None:
        super().__init__()
        self.fc = LorentzFC(
            in_features=hidden_dim + 1, # lorentz dim   
            out_features=hidden_dim + 1, # lorentz dim   
            manifold=manifold,
            reset_params=reset_params,
            a_default=a_default,
            do_mlr=False
        )

    def forward(self, x_lorentz: torch.Tensor) -> torch.Tensor:
        return self.fc(x_lorentz)


class PersonalMLPHead(nn.Module):
    def __init__(self, hidden_dim: int, manifold: Lorentz, num_classes: int, curvature: float, reset_params: str = "lorentz_kaiming", a_default: float = 0.0) -> None:
        super().__init__()
        self.fc = LorentzFC(
            in_features=hidden_dim + 1, # lorentz dim
            out_features=num_classes + 1, # lorentz dim
            manifold=manifold,
            reset_params=reset_params,
            a_default=a_default,
            do_mlr=True
        )

    def forward(self, x_lorentz: torch.Tensor) -> torch.Tensor:
        return self.fc(x_lorentz)


class PersonalModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        num_heads: int = 1,
        compute_scores: str = "lorentz_inner",
        head_fusion: str = "midpoint",
        split_heads: bool | None = True,
        curvature: float = 1.0,
        attn_mask: torch.Tensor = None,
        a_default: float = 0.0,
        reset_params: str = "lorentz_kaiming",
        dropout: float = 0.0,
        use_ffn: bool = False,
        train_curvature: bool = False
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.use_ffn = use_ffn
        self.dropout = nn.Dropout(dropout)
        self.train_curvature = train_curvature
        
        self.manifold = Lorentz(curvature)

        self.lin_in = nn.Linear(input_dim, hidden_dim)

        self.mha_layers = nn.ModuleList([
            PersonalMHA(
                hidden_dim=hidden_dim,
                manifold=self.manifold,
                curvature=curvature,
                num_heads=num_heads,
                compute_scores=compute_scores,
                head_fusion=head_fusion,
                split_heads=split_heads,
                a_default=a_default,
                reset_params=reset_params
            )
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            PersonalFFN(
                hidden_dim=hidden_dim,
                manifold=self.manifold,
                curvature=curvature,
                reset_params=reset_params,
                a_default=a_default
            )
            for _ in range(num_layers)
        ])

        self.head = PersonalMLPHead(
            hidden_dim=hidden_dim,
            manifold=self.manifold,
            num_classes=num_classes,
            curvature=curvature,
            reset_params=reset_params,
            a_default=a_default
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

        # project to hidden dim and dropout
        x_space = self.lin_in(x)  
        x_space = self.dropout(x_space)
        x_space = F.normalize(x_space, p=2, dim=-1) * 1.0

        # map to Lorentz manifold
        x_lorentz = self.manifold.expmap0(x_space)  

        for mha, ffn in zip(self.mha_layers, self.ffn_layers):

            x_lorentz = mha(x_lorentz, attn_mask=self.attn_mask)

            if self.use_ffn:
                x_lorentz = ffn(x_lorentz)

        logits = self.head(x_lorentz)

        if squeeze_batch:
            logits = logits.squeeze(0)
        return logits
