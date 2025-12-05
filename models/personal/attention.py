import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma

from manifolds.personal import Lorentz, LorentzFC

logger = logging.getLogger(__name__)

class LorentzAttention(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            curvature: float = 0.1, 
            num_heads: int = 1, 
            compute_scores: str = "lorentz_inner", 
            value_agg: str = "riemannian", 
            concat_operation: str = "direct", 
            out_dim: int | None = None,
            a_default: float = 0.0,
            split_qkv: bool = False,
            debug: bool = False
        ) -> None:
        super().__init__()
        self.lorentz_dim = input_dim
        self.curvature = curvature
        self.num_heads = num_heads
        self.compute_scores = compute_scores
        self.value_agg = value_agg
        self.concat_operation = concat_operation
        self.split_qkv = split_qkv
        self.debug = debug

        self.manifold = Lorentz(self.curvature)

        self.spatial_dim = self.lorentz_dim - 1
        assert self.spatial_dim % num_heads == 0
        self.head_spatial_dim = self.spatial_dim // num_heads
        self.head_lorentz_dim = self.head_spatial_dim + 1       # +1 for time component

        self.scale = 1.0 / math.sqrt(self.head_spatial_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

        if self.split_qkv:
            self.W_q = LorentzFC(self.spatial_dim, self.num_heads * self.head_spatial_dim, manifold=self.manifold)
            self.W_k = LorentzFC(self.spatial_dim, self.num_heads * self.head_spatial_dim, manifold=self.manifold)
            self.W_v = LorentzFC(self.spatial_dim, self.num_heads * self.head_spatial_dim, manifold=self.manifold)
        else:
            self.W_k = nn.ModuleList()
            self.W_q = nn.ModuleList()
            self.W_v = nn.ModuleList()
            for _ in range(self.num_heads):
                self.W_k.append(LorentzFC(self.spatial_dim, self.head_spatial_dim, manifold=self.manifold))
                self.W_q.append(LorentzFC(self.spatial_dim, self.head_spatial_dim, manifold=self.manifold))
                self.W_v.append(LorentzFC(self.spatial_dim, self.head_spatial_dim, manifold=self.manifold))

        in_spatial_out = self.num_heads * self.head_spatial_dim
        if out_dim is None:
            out_dim = self.spatial_dim
        self.out_dim = out_dim  

        self.W_o = LorentzFC(in_spatial_out, self.out_dim, manifold=self.manifold)

        if self.concat_operation == "log-radius":
            n = self.num_heads * self.head_spatial_dim          # total spatial dim after concat
            ni = self.head_spatial_dim                          # spatial dim per head

            s_val = torch.exp(0.5 * (digamma(torch.tensor(n / 2.0)) - digamma(torch.tensor(ni / 2.0))))
            self.register_buffer("log_radius_scale", s_val)     # scalar buffer
        
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
        k = -self.curvature                     # k > 0 if curvature < 0
        d2 = -2.0 * k - 2.0 * lp
        return d2

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, 1 + H * d_head] (on manifold already)
        return: [B, H, N, 1 + d_head], preserving the original time component
        """
        B, N, D = x.shape
        H = self.num_heads
        d_head = self.head_spatial_dim  # spatial per head

        assert D == 1 + H * d_head

        time = x[..., :1]                         # [B, N, 1]
        space = x[..., 1:]                        # [B, N, H * d_head]

        space = space.view(B, N, H, d_head)       # [B, N, H, d_head]

        # replicate time across heads (same time for all heads)
        time = time.unsqueeze(2).expand(-1, -1, H, -1)  # [B, N, H, 1]

        out = torch.cat([time, space], dim=-1)    # [B, N, H, 1 + d_head]
        out = out.permute(0, 2, 1, 3).contiguous()     # [B, H, N, 1 + d_head]

        return out

    def _concat_heads_direct(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D_head = x.shape
        d_head = D_head - 1

        x = x.permute(0, 2, 1, 3)       # [B, N, H, 1 + d_head]
        time = x[..., :1]               # [B, N, H, 1]
        space = x[..., 1:]              # [B, N, H, d_head]

        space_cat = space.reshape(B, N, H * d_head)        # [B, N, H * d_head]

        k = self.manifold.k()
        u_sq = (space_cat ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        t_prime_sq = 1.0 / k + u_sq
        t_prime = torch.sqrt(torch.clamp(t_prime_sq, min=1e-9))

        out = torch.cat([t_prime, space_cat], dim=-1)      # [B, N, 1 + H * d_head]
        return out
    
    def _concat_heads_log_radius(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D_h = x.shape
        d_head = D_h - 1

        x = x.permute(0, 2, 1, 3)                          # [B, N, H, 1 + d_head]
        time = x[..., :1]                                  # [B, N, H, 1]
        space = x[..., 1:]                                 # [B, N, H, d_head]

        k_val = self.manifold.k()                          # > 0
        s = self.log_radius_scale                          # scalar tensor

        space_tilde = s * space                            # [B, N, H, d_head]

        t_sq = (time ** 2).squeeze(-1)                     # [B, N, H]
        sum_term = (t_sq - 1.0 / k_val).sum(dim=2, keepdim=True)  # [B, N, 1]
        t_prime_sq = 1.0 / k_val + (s ** 2) * sum_term
        t_prime = torch.sqrt(torch.clamp(t_prime_sq, min=1e-9))   # [B, N, 1]

        space_cat = space_tilde.reshape(B, N, H * d_head)  # [B, N, H * d_head]

        out = torch.cat([t_prime, space_cat], dim=-1)      # [B, N, 1 + H * d_head]
        return out

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        assert D == self.lorentz_dim, f"Expected last dim {self.lorentz_dim}, got {D}"

        if self.debug:
            logger.debug("====== LorentzAttention forward ======")
            logger.debug(
                "Input: on_manifold=%s | has_nans=%s",
                self._is_on_manifold(x, self.manifold, log_details=False),
                bool(torch.isnan(x).any()),
            )

        if self.split_qkv:
            q_big = self.W_q(x)                 # [B, N, 1 + H * head_spatial_dim]
            k_big = self.W_k(x)
            v_big = self.W_v(x)

            q = self._split_heads(q_big)        # [B, H, N, 1 + d_head]
            k = self._split_heads(k_big)
            v = self._split_heads(v_big)
        else:
            q_heads, k_heads, v_heads = [], [], []
            for h in range(self.num_heads):
                q_heads.append(self.W_q[h](x))  # [B, N, 1 + d_head]
                k_heads.append(self.W_k[h](x))
                v_heads.append(self.W_v[h](x))

            q = torch.stack(q_heads, dim=1)     # [B, H, N, 1 + d_head]
            k = torch.stack(k_heads, dim=1)
            v = torch.stack(v_heads, dim=1)
        
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

        if mask is not None:
            mask_exp = mask[:, None, None, :].to(dtype=torch.bool)  # [B, 1, 1, N]
            scores = scores.masked_fill(~mask_exp, float("-inf"))

        attn = F.softmax(scores, dim=-1)    # [B, H, N, N] 

        if self.debug:
            logger.debug(
                "Attn: has_nans=%s | min=%.3e | max=%.3e",
                bool(torch.isnan(attn).any()),
                attn.min().item(),
                attn.max().item(),
            )

        if self.value_agg == "riemannian": #NOTE: doesn't work for multiple heads
            v_tan = self.manifold.logmap0(v)
            out_tan = torch.einsum("bhnm,bhmd->bhnd", attn, v_tan)
            out = self.manifold.expmap0(out_tan)
        elif self.value_agg == "midpoint":
            out = self.manifold.mid_point(v, attn)
        else:
            raise ValueError(f"Unknown value_agg mode: {self.value_agg}")
        
        if self.debug:
            logger.debug(
                "Attn output: on_manifold=%s | has_nans=%s",
                self._is_on_manifold(out, self.manifold, log_details=False),
                bool(torch.isnan(out).any()),
            )
                    
        if self.num_heads > 1:
            if self.concat_operation == "direct":
                out = self._concat_heads_direct(out)
            elif self.concat_operation == "log-radius":
                out = self._concat_heads_log_radius(out)
            else:
                raise NotImplementedError(f"Unknown concat_operation: {self.concat_operation}")
        else:
            out = out.squeeze(1)    # [B, N, 1 + head_spatial_dim]

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
    
