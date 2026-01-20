import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma

from models.personal.lorentz import Lorentz
from models.personal.layer import LorentzFC

logger = logging.getLogger(__name__)

class LorentzAttention(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            curvature: float = 0.1, 
            num_heads: int = 1, 
            compute_scores: str = "lorentz_inner", 
            head_fusion: str = "midpoint",
            split_heads: bool | None = True,
            out_dim: int | None = None,
            reset_params: str = "lorentz_kaiming",
            a_default: float = 0.0,
            attn_mask: torch.Tensor | None = None
        ) -> None:
        super().__init__()
        self.lorentz_dim = input_dim
        self.spatial_dim = input_dim - 1
        self.curvature = curvature
        self.num_heads = num_heads
        self.compute_scores = compute_scores
        self.head_fusion = head_fusion
        self.attn_mask = attn_mask 

        self.manifold = Lorentz(curvature)

        if split_heads is None:
            split_heads = (head_fusion != "midpoint")  # midpoint => HypFormer-style full-dim heads
        if head_fusion == "midpoint" and split_heads:
            raise ValueError("head_fusion='midpoint' does not support split_heads=True (would shrink dim to D/H).")
        if head_fusion in ("concat_direct", "concat_logradius") and not split_heads:
            raise ValueError("concat_* head_fusion requires split_heads=True.")
        self.split_heads = split_heads

        if self.num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if self.split_heads:
            if self.spatial_dim % self.num_heads != 0:
                raise ValueError(f"spatial_dim ({self.spatial_dim}) must be divisible by num_heads ({self.num_heads})")
            self.head_spatial_dim = self.spatial_dim // self.num_heads
        else:
            self.head_spatial_dim = self.spatial_dim  # full-dim heads

        self.head_lorentz_dim = self.head_spatial_dim + 1
        self.scale = 1.0 / math.sqrt(self.head_spatial_dim)

        self.W_q = nn.ModuleList([
            LorentzFC(
                in_features=self.lorentz_dim, 
                out_features=self.head_lorentz_dim, 
                manifold=self.manifold, 
                reset_params="lorentz_kaiming", 
                a_default=a_default)
            for _ in range(self.num_heads)
        ])
        self.W_k = nn.ModuleList([
            LorentzFC(
                in_features=self.lorentz_dim, 
                out_features=self.head_lorentz_dim,
                manifold=self.manifold, 
                reset_params="lorentz_kaiming", 
                a_default=a_default)
            for _ in range(self.num_heads)
        ])
        self.W_v = nn.ModuleList([
            LorentzFC(
                in_features=self.lorentz_dim, 
                out_features=self.head_lorentz_dim, 
                manifold=self.manifold, 
                reset_params="lorentz_kaiming", 
                a_default=a_default)
            for _ in range(self.num_heads)
        ])

        # output projection expects different spatial size depending on fusion
        if self.num_heads == 1:
            in_spatial_out = self.head_spatial_dim
        else:
            if self.head_fusion == "midpoint":
                in_spatial_out = self.spatial_dim  # full-dim heads; midpoint keeps D
            else:
                in_spatial_out = self.spatial_dim  # concat_* reconstructs full D

        if out_dim is None:
            out_dim = self.spatial_dim
        self.out_dim = out_dim

        self.W_o = LorentzFC(
            in_features=in_spatial_out + 1, 
            out_features=self.out_dim + 1, 
            manifold=self.manifold, 
            reset_params="lorentz_kaiming",
            a_default=a_default
        ) 
        
        if self.head_fusion == "concat_logradius":
            n = self.spatial_dim            
            ni = self.head_spatial_dim
            s_val = torch.exp(
                0.5 * (digamma(torch.tensor(n / 2.0)) - digamma(torch.tensor(ni / 2.0)))
            )
            self.register_buffer("log_radius_scale", s_val)
        
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

        if log_details and self.debug:
            logger.debug("[manifold check] k=%.6f", k_val)
            logger.debug("  target Minkowski norm = %.6f", target)
            logger.debug("  max |<x,x>_L - target| = %.6e", max_diff)

        return max_diff < tol
        
    @staticmethod
    def _lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Lorentz inner product: -t_x * t_y + <s_x, s_y>
        """
        x_t, x_x = x[..., :1], x[..., 1:]  # [B, H, N, 1], [B, H, N, head_spatial_dim]
        y_t, y_x = y[..., :1], y[..., 1:]  # [B, H, N, 1], [B, H, N, head_spatial_dim]

        time = torch.matmul(x_t, y_t.transpose(-1, -2))    # [B, H, N, 1] @ [B, H, 1, N] -> [B, H, N, N]
        space = torch.matmul(x_x, y_x.transpose(-1, -2))   # [B, H, N, head_spatial_dim] @ [B, H, head_spatial_dim, N] -> [B, H, N, N]

        return -time + space  # [B, H, N, N]

    def _lorentz_sqdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lp = self._lorentz_inner(x, y)          # [B, H, N, N]
        k = self.manifold.k()
        d2 = -2.0 * k - 2.0 * lp  # [B, H, N, N]
        return d2

    def _concat_heads_direct(self, x: torch.Tensor) -> torch.Tensor:
        # Qu, E., et al. (https://openreview.net/forum?id=NQi9U0YLW3)
        B, H, N, Dh = x.shape
        d_head = Dh - 1
        k = torch.as_tensor(self.manifold.k(), device=x.device, dtype=x.dtype)

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, H, N, 1+head_spatial_dim] -> [B, N, H, 1+head_spatial_dim]
        time = x[..., :1]                       # [B, N, H, 1]
        space = x[..., 1:]                      # [B, N, H, head_spatial_dim]

        space_cat = space.reshape(B, N, H * d_head)  # [B, N, H*head_spatial_dim] = [B, N, spatial_dim]

        # t'² = Σt_i² + (H-1)/k
        time_sq_sum = (time ** 2).sum(dim=2)                # [B, N, H, 1] -> [B, N, 1]
        time_sq_sum = time_sq_sum + (H - 1) / k  # [B, N, 1]
        t_prime = torch.sqrt(time_sq_sum.clamp_min(1e-9))   # [B, N, 1]

        out = torch.cat([t_prime, space_cat], dim=-1)  # [B, N, 1+spatial_dim]
        return out
    
    def _concat_heads_log_radius(self, x: torch.Tensor) -> torch.Tensor:
        # Anonymous, et al. (https://openreview.net/forum?id=NNnkLi1ALt)
        B, H, N, Dh = x.shape
        d_head = Dh - 1
        k = torch.as_tensor(self.manifold.k(), device=x.device, dtype=x.dtype)

        s = self.log_radius_scale.to(device=x.device, dtype=x.dtype)  # scalar

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, H, N, 1+head_spatial_dim] -> [B, N, H, 1+head_spatial_dim]
        time = x[..., :1]                       # [B, N, H, 1]
        space = x[..., 1:]                      # [B, N, H, head_spatial_dim]

        # scale spatial components: ũi = s * ui
        space_scaled = (s * space)  # [B, N, H, head_spatial_dim]
        space_cat = space_scaled.reshape(B, N, H * d_head)  # [B, N, H*head_spatial_dim] = [B, N, spatial_dim]

        # recompute time coordinate: t' = sqrt(1/K + s² * Σ(ti² + 1/K))
        time_sq_plus_invk = (time ** 2) + (1.0 / k)  # [B, N, H, 1]
        time_sum = time_sq_plus_invk.sum(dim=2)      # [B, N, H, 1] -> [B, N, 1]
        t_prime = torch.sqrt((1.0 / k + s ** 2 * time_sum).clamp_min(1e-9))  # [B, N, 1]

        out = torch.cat([t_prime, space_cat], dim=-1)  # [B, N, 1+spatial_dim]
        return out

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        assert D == self.lorentz_dim, f"Expected last dim {self.lorentz_dim}, got {D}"

        # project to per-head dimensions: [B, N, 1+spatial_dim] -> [B, N, 1+head_spatial_dim] per head
        q = torch.stack([self.W_q[h](x) for h in range(self.num_heads)], dim=1)  # [B, H, N, 1+head_spatial_dim]
        k = torch.stack([self.W_k[h](x) for h in range(self.num_heads)], dim=1)  # [B, H, N, 1+head_spatial_dim]
        v = torch.stack([self.W_v[h](x) for h in range(self.num_heads)], dim=1)  # [B, H, N, 1+head_spatial_dim]
        
        if self.compute_scores == "lorentz_inner":
            scores = self._lorentz_inner(q, k) * self.scale   # [B, H, N, N]
        elif self.compute_scores == "signed_dist":
            d2 = self._lorentz_sqdist(q, k)             # [B, H, N, N]
            scores = -d2 * self.scale                   # [B, H, N, N]
        else:
            raise ValueError(f"Unknown compute_scores mode: {self.compute_scores}")
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                m = attn_mask[None, None, :, :]      # [N, N] -> [1, 1, N, N]
            elif attn_mask.dim() == 3:
                m = attn_mask[:, None, :, :]         # [B, N, N] -> [B, 1, N, N]
            else:
                raise ValueError(f"attn_mask must be 2D or 3D, got {attn_mask.dim()}D")
            scores = scores.masked_fill(~m, -1e9)  # [B, H, N, N]

        attn = F.softmax(scores, dim=-1)  # [B, H, N, N]

        # weighted aggregation: [B, H, N, 1+head_spatial_dim] with weights [B, H, N, N]
        # mid_point computes: weighted_sum = attn @ v, then projects to manifold
        out = self.manifold.lorentz_midpoint(v, attn)  # [B, H, N, 1+head_spatial_dim]
        
        # fuse multiple heads
        if self.num_heads > 1:
            if self.head_fusion == "midpoint":
                B, H, N, D_lorentz = out.shape
                out = out.permute(0, 2, 1, 3).contiguous().view(B * N, H, D_lorentz) # [B, N, H, D_lorentz] -> [B*N, H, D_lorentz]
                out = self.manifold.lorentz_midpoint(out).view(B, N, D_lorentz) # [B, N, D_lorentz] -> [B, N, 1+spatial_dim]
            elif self.head_fusion == "concat_direct":
                out = self._concat_heads_direct(out)  # [B, H, N, 1+head_spatial_dim] -> [B, N, 1+spatial_dim]
            elif self.head_fusion == "concat_logradius":
                out = self._concat_heads_log_radius(out)  # [B, H, N, 1+head_spatial_dim] -> [B, N, 1+spatial_dim]
        else:
            out = out.squeeze(1)  # [B, 1, N, 1+head_spatial_dim] -> [B, N, 1+head_spatial_dim]

        # output projection: [B, N, 1+spatial_dim] -> [B, N, 1+out_dim]
        out = self.W_o(out) 
        
        return out 
    
