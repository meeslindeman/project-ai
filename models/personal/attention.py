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
            concat_operation: str = "direct", 
            out_dim: int | None = None,
            a_default: float = 0.0,
            split_heads: bool = True,
            debug: bool = False,
            attn_mask: torch.Tensor | None = None
        ) -> None:
        super().__init__()
        self.lorentz_dim = input_dim
        self.curvature = curvature
        self.num_heads = num_heads
        self.compute_scores = compute_scores
        self.concat_operation = concat_operation
        self.split_heads = split_heads
        self.debug = debug

        self.manifold = Lorentz(self.curvature)
        self.spatial_dim = self.lorentz_dim - 1

        if self.split_heads:
            assert self.spatial_dim % num_heads == 0
            self.head_spatial_dim = self.spatial_dim // num_heads
        else:
            # hypformer-like heads: each head has full spatial dim
            self.head_spatial_dim = self.spatial_dim 

        self.head_lorentz_dim = self.head_spatial_dim + 1
        self.scale = 1.0 / math.sqrt(self.head_spatial_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # validate head fusion choices
        if (not self.split_heads) and (self.concat_operation in ("direct", "log-radius")):
            raise ValueError("direct/log-radius concat requires split_heads=True (D//H per head).")
        if self.num_heads > 1 and self.concat_operation == "none" and self.split_heads:
            raise ValueError("concat_operation='none' with split_heads=True shrinks dim; set split_heads=False for HypFormer-like.")

        self.W_q = nn.ModuleList()
        self.W_k = nn.ModuleList()
        self.W_v = nn.ModuleList()
        head_out = self.head_spatial_dim
        for _ in range(self.num_heads):
            self.W_q.append(LorentzFC(self.spatial_dim, head_out, manifold=self.manifold))
            self.W_k.append(LorentzFC(self.spatial_dim, head_out, manifold=self.manifold))
            self.W_v.append(LorentzFC(self.spatial_dim, head_out, manifold=self.manifold))

        # output projection expects different spatial size depending on fusion
        if self.num_heads == 1 or self.concat_operation == "none":
            in_spatial_out = self.head_spatial_dim
        else:
            in_spatial_out = self.num_heads * self.head_spatial_dim

        if out_dim is None:
            out_dim = self.spatial_dim
        self.out_dim = out_dim

        self.W_o = LorentzFC(in_spatial_out, self.out_dim, manifold=self.manifold)

        if self.concat_operation == "log-radius":
            n = self.num_heads * self.head_spatial_dim
            ni = self.head_spatial_dim
            s_val = torch.exp(0.5 * (digamma(torch.tensor(n / 2.0)) - digamma(torch.tensor(ni / 2.0))))
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
        B, H, N, D_head = x.shape
        d_head = D_head - 1
        x = x.permute(0, 2, 1, 3)          # [B,N,H,1+d_head]
        space = x[..., 1:]                 # [B,N,H,d_head]
        space_cat = space.reshape(B, N, H * d_head)

        k = self.manifold.k()
        u_sq = (space_cat ** 2).sum(dim=-1, keepdim=True)
        t_prime = torch.sqrt(torch.clamp(1.0 / k + u_sq, min=1e-9))
        out = torch.cat([t_prime, space_cat], dim=-1)
        return out
    
    def _concat_heads_log_radius(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D_h = x.shape
        d_head = D_h - 1

        x = x.permute(0, 2, 1, 3)          # [B,N,H,1+d_head]
        time = x[..., :1]                  # [B,N,H,1]
        space = x[..., 1:]                 # [B,N,H,d_head]

        k_val = self.manifold.k()
        s = self.log_radius_scale
        space_tilde = s * space

        t_sq = (time ** 2).squeeze(-1)     # [B,N,H]
        sum_term = (t_sq - 1.0 / k_val).sum(dim=2, keepdim=True)
        t_prime = torch.sqrt(torch.clamp(1.0 / k_val + (s ** 2) * sum_term, min=1e-9))

        space_cat = space_tilde.reshape(B, N, H * d_head)
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
            m = attn_mask[None, None, :, :]  # [1,1,N,N]
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
        # omit riemannian for now, does not work well with multiple heads
        out = self.manifold.mid_point(v, attn)
        
        if self.debug:
            logger.debug(
                "Attn output: on_manifold=%s | has_nans=%s",
                self._is_on_manifold(out, self.manifold, log_details=False),
                bool(torch.isnan(out).any()),
            )
                    
        if self.num_heads > 1:
            if self.concat_operation == "none":
                # explicit midpoint over heads for each (B,N)
                B, H, N, Dlor = out.shape
                out = out.permute(0, 2, 1, 3).contiguous().view(B * N, H, Dlor)
                out = self.manifold.mid_point(out).view(B, N, Dlor)
            elif self.concat_operation == "direct":
                out = self._concat_heads_direct(out)      # [B,N,1+D]
            elif self.concat_operation == "log-radius":
                out = self._concat_heads_log_radius(out)  # [B,N,1+D]
            else:
                raise NotImplementedError(f"Unknown concat_operation: {self.concat_operation}")
        else:
            out = out.squeeze(1)  # [B,N,1+d_head]

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
    
