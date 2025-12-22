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
            a_default: float = 0.0,
            debug: bool = False,
            attn_mask: torch.Tensor | None = None
        ) -> None:
        super().__init__()
        self.lorentz_dim = input_dim
        self.spatial_dim = input_dim - 1
        self.curvature = curvature
        self.num_heads = num_heads
        self.compute_scores = compute_scores
        self.head_fusion = head_fusion
        self.debug = debug
        self.attn_mask = attn_mask 

        self.manifold = Lorentz(curvature)

        valid_fusions = ("midpoint", "concat_direct", "concat_logradius")
        if head_fusion not in valid_fusions:
            raise ValueError(f"Unknown head_fusion: {head_fusion}. Must be one of {valid_fusions}")

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
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.W_q = nn.ModuleList([
            LorentzFC(self.spatial_dim, self.head_spatial_dim, manifold=self.manifold)
            for _ in range(self.num_heads)
        ])
        self.W_k = nn.ModuleList([
            LorentzFC(self.spatial_dim, self.head_spatial_dim, manifold=self.manifold)
            for _ in range(self.num_heads)
        ])
        self.W_v = nn.ModuleList([
            LorentzFC(self.spatial_dim, self.head_spatial_dim, manifold=self.manifold)
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
        self.W_o = LorentzFC(in_spatial_out, self.out_dim, manifold=self.manifold)

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
        x_t, x_x = x[..., :1], x[..., 1:]
        y_t, y_x = y[..., :1], y[..., 1:]

        time = torch.matmul(x_t, y_t.transpose(-1, -2))    # [B, H, N, N]
        space = torch.matmul(x_x, y_x.transpose(-1, -2))   # [B, H, N, N]

        return -time + space

    def _lorentz_sqdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lp = self._lorentz_inner(x, y)          # [B, H, N, N]
        k = self.manifold.k()
        d2 = -2.0 * k - 2.0 * lp
        return d2

    def _concat_heads_direct(self, x: torch.Tensor) -> torch.Tensor:
        # Qu, E., et al. (https://openreview.net/forum?id=NQi9U0YLW3)
        B, H, N, Dh = x.shape
        d_head = Dh - 1
        k = torch.as_tensor(self.manifold.k(), device=x.device, dtype=x.dtype)

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, N, H, Dh]
        time = x[..., :1]                       # [B, N, H, 1]
        space = x[..., 1:]                      # [B, N, H, d_head]

        # concatenate spatial components 
        space_cat = space.reshape(B, N, H * d_head)

        # t'^2 = sum_i t_i^2 - (H-1)/k
        time_sq_sum = (time ** 2).sum(dim=2)          # [B, N, 1]
        t_prime = torch.sqrt((time_sq_sum - (H - 1) / k).clamp_min(1e-9))  # [B, N, 1]

        out = torch.cat([t_prime, space_cat], dim=-1)
        return out
    
    def _concat_heads_log_radius(self, x: torch.Tensor) -> torch.Tensor:
        # Intrinci Lorentz paper
        B, H, N, Dh = x.shape
        d_head = Dh - 1
        k = torch.as_tensor(self.manifold.k(), device=x.device, dtype=x.dtype)

        s = self.log_radius_scale.to(device=x.device, dtype=x.dtype)

        x = x.permute(0, 2, 1, 3).contiguous()  
        space = x[..., 1:]                       # [B, N, H, d_head]

        # scale spatial components
        space_cat = (s * space).reshape(B, N, H * d_head)

        # recompute a single time coordinate so that:
        # -t'^2 + ||space_cat||^2 = -1/k  =>  t'^2 = 1/k + ||space_cat||^2
        space_sq = (space_cat ** 2).sum(dim=-1, keepdim=True)       # [B, N, 1]
        t_prime = torch.sqrt((1.0 / k + space_sq).clamp_min(1e-9))  # [B, N, 1]

        out = torch.cat([t_prime, space_cat], dim=-1)
        return out

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        assert D == self.lorentz_dim, f"Expected last dim {self.lorentz_dim}, got {D}"

        if self.debug:
            logger.debug("====== LorentzAttention forward ======")
            logger.debug(
                "Input: on_manifold=%s | has_nans=%s",
                self._is_on_manifold(x, self.manifold, log_details=False),
                bool(torch.isnan(x).any()),
            )

        q = torch.stack([self.W_q[h](x) for h in range(self.num_heads)], dim=1)  # [B,H,N,1+d_head]
        k = torch.stack([self.W_k[h](x) for h in range(self.num_heads)], dim=1)
        v = torch.stack([self.W_v[h](x) for h in range(self.num_heads)], dim=1)
        
        if self.debug:
            logger.debug(
                "Q: on_manifold=%s | has_nans=%s",
                self._is_on_manifold(q, self.manifold, log_details=False),
                bool(torch.isnan(q).any()),
            )
            logger.debug(
                "K: on_manifold=%s | has_nans=%s",
                self._is_on_manifold(k, self.manifold, log_details=False),
                bool(torch.isnan(k).any()),
            )
            logger.debug(
                "V: on_manifold=%s | has_nans=%s",
                self._is_on_manifold(v, self.manifold, log_details=False),
                bool(torch.isnan(v).any()),
            )

        if self.compute_scores == "lorentz_inner":
            scores = self._lorentz_inner(q, k) * self.scale   
        elif self.compute_scores == "signed_dist":
            d2 = self._lorentz_sqdist(q, k)             
            scores = -d2 * self.scale * self.temperature
        else:
            raise ValueError(f"Unknown compute_scores mode: {self.compute_scores}")
        
        if self.debug:
            logger.debug(
                "Scores: has_nans=%s | min=%.3e | max=%.3e",
                bool(torch.isnan(scores).any()),
                scores.min().item(),
                scores.max().item(),
            )

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                m = attn_mask[None, None, :, :]      # [1,1,N,N]
            elif attn_mask.dim() == 3:
                m = attn_mask[:, None, :, :]         # [B,1,N,N]
            else:
                raise ValueError(...)
            scores = scores.masked_fill(~m, -1e9)

        attn = F.softmax(scores, dim=-1)  # [B,H,N,N]

        if self.debug:
            logger.debug(
                "Attn: has_nans=%s | min=%.3e | max=%.3e",
                bool(torch.isnan(attn).any()),
                attn.min().item(),
                attn.max().item(),
            )

        # use mid_point only
        # optional: look at gyro agg
        out = self.manifold.mid_point(v, attn)
        
        if self.debug:
            logger.debug(
                "Attn output: on_manifold=%s | has_nans=%s",
                self._is_on_manifold(out, self.manifold, log_details=False),
                bool(torch.isnan(out).any()),
            )
                    
        if self.num_heads > 1:
            if self.head_fusion == "midpoint":
                # hypformer style -> full-dim heads + midpoint
                B, H, N, D_lorentz = out.shape
                out = out.permute(0, 2, 1, 3).contiguous().view(B * N, H, D_lorentz)
                out = self.manifold.mid_point(out).view(B, N, D_lorentz)
            elif self.head_fusion == "concat_direct":
                out = self._concat_heads_direct(out)      
            elif self.head_fusion == "concat_logradius":
                out = self._concat_heads_log_radius(out) 
        else:
            out = out.squeeze(1)

        if self.debug:
            logger.debug(
                "After heads concat: on_manifold=%s | has_nans=%s",
                self._is_on_manifold(out, self.manifold, log_details=False),
                bool(torch.isnan(out).any()),
            )

        out = self.W_o(out)
        
        if self.debug:
            logger.debug(
                "Final output: on_manifold=%s | has_nans=%s",
                self._is_on_manifold(out, self.manifold, log_details=False),
                bool(torch.isnan(out).any()),
            )
            logger.debug("=======================================")
            
        return out
    
