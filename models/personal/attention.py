import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma

from manifolds.personal import Lorentz, LorentzFC


class LorentzAttention(nn.Module):
    def __init__(self, input_dim: int, curvature: float = 0.1, num_heads: int = 1, heads_concat: bool = True, compute_scores: str = "lorentz_inner", value_agg: str = "riemannian", concat_operation: str = "direct", out_dim: int | None = None) -> None:
        super().__init__()
        self.lorentz_dim = input_dim
        self.curvature = curvature    
        self.num_heads = num_heads
        self.heads_concat = heads_concat
        self.compute_scores = compute_scores
        self.value_agg = value_agg
        self.concat_operation = concat_operation

        self.manifold = Lorentz(self.curvature)

        self.spatial_dim = self.lorentz_dim - 1
        assert self.spatial_dim % num_heads == 0
        self.head_spatial_dim = self.spatial_dim // num_heads
        self.head_lorentz_dim = self.head_spatial_dim + 1 # +1 for time component

        self.scale = 1.0 / math.sqrt(self.head_spatial_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))
 
        # Q, K, V projections on the manifold
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

        # Log-radius concat costants
        if self.concat_operation == "log-radius":
            n = self.num_heads * self.head_spatial_dim # total spatial dim after concat
            ni = self.head_spatial_dim # spatial dim per head

            # s(n, ni) = exp(0.5 * (digamma(n/2) - digamma(ni/2)))
            s_val = torch.exp(0.5 * (digamma(torch.tensor(n / 2.0)) - digamma(torch.tensor(ni / 2.0))))
            self.register_buffer("log_radius_scale", s_val) # scalar buffer
    
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
    
    def _concat_heads_direct(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D_head = x.shape
        d_head = D_head - 1
        
        # Move heads next to feature dim: [B, N, H, 1+ d_head]
        x = x.permute(0, 2, 1, 3)       # [B, N, H, 1 + d_head]
        time = x[..., :1]               # [B, N, H, 1]
        space = x[..., 1:]              # [B, N, H, d_head]

        # Concatenate all spatial parts: [B, N, H * d_head]
        space_cat = space.reshape(B, N, H * d_head)

        # Compute new time component so that -t'^2 + ||u_cat||^2 = -1/k
        k = self.manifold.k()
        u_sq = (space_cat ** 2).sum(dim=-1, keepdim=True)       # [B, N, 1]
        t_prime_sq = 1.0 / k + u_sq
        t_prime = torch.sqrt(torch.clamp(t_prime_sq, min=1e-9))

        out = torch.cat([t_prime, space_cat], dim=-1)           # [B, N, 1 + H * d_head]
        return out
    
    def _concat_heads_log_radius(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D_h = x.shape
        d_head = D_h - 1

        # [B, N, H, 1+d_head]
        x = x.permute(0, 2, 1, 3)
        time = x[..., :1]                                      # [B, N, H, 1]
        space = x[..., 1:]                                     # [B, N, H, d_head]

        k_val = self.manifold.k()                                   # > 0
        s = self.log_radius_scale                                   # scalar tensor

        # Scale spatial parts: space_tilde = s * space
        space_tilde = s * space                                     # [B, N, H, d_head]

        # Compute t'^2 = 1/k + s^2 * sum_h (t_h^2 - 1/k)
        t_sq = (time ** 2).squeeze(-1)                              # [B, N, H]
        sum_term = (t_sq - 1.0 / k_val).sum(dim=2, keepdim=True)    # [B, N, 1]
        t_prime_sq = 1.0 / k_val + (s ** 2) * sum_term
        t_prime = torch.sqrt(torch.clamp(t_prime_sq, min=1e-9))     # [B, N, 1]

        # Concatenate scaled spatial parts across heads
        space_cat = space_tilde.reshape(B, N, H * d_head)           # [B, N, H*d_head]

        out = torch.cat([t_prime, space_cat], dim=-1)               # [B, N, 1 + H*d_head]
        return out


    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        assert D == self.lorentz_dim, f"Expected last dim {self.lorentz_dim}, got {D}"

        # Compute Q, K, V for each head
        q_heads, k_heads, v_heads = [], [], []
        for h in range(self.num_heads):
            q_heads.append(self.W_q[h](x))  # [B, N, head_lorentz_dim]
            k_heads.append(self.W_k[h](x))  # [B, N, head_lorentz_dim]
            v_heads.append(self.W_v[h](x))  # [B, N, head_lorentz_dim]

        # Stack into [B, H, N, head_lorentz_dim]
        q = torch.stack(q_heads, dim=1)
        k = torch.stack(k_heads, dim=1)
        v = torch.stack(v_heads, dim=1)

        if self.compute_scores == "lorentz_inner":
            # Plain Lorentz inner product scores
            scores = self._lorentz_inner(q, k) * self.scale   
        elif self.compute_scores == "signed_dist":
            # Scaled negative squared distance
            d2 = self._lorentz_sqdist(q, k)             
            scores = -d2 * self.scale * self.temperature
        else:
            raise ValueError(f"Unknown compute_scores mode: {self.compute_scores}")

        if mask is not None:
            # mask: [B, N] -> [B, 1, 1, N]
            mask_exp = mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~mask_exp, float("-inf"))

        attn = F.softmax(scores, dim=-1)    # [B, H, N, N]

        if self.value_agg == "ambient":
            v_space = v[..., 1:] 
            out_space = torch.einsum("bhnm,bhmd->bhnd", attn, v_space)
            out = self.manifold.projection_space_orthogonal(out_space)
        elif self.value_agg == "riemannian":
            v_tan = self.manifold.logmap0(v)
            out_tan = torch.einsum("bhnm,bhmd->bhnd", attn, v_tan)
            out = self.manifold.expmap0(out_tan)
        else:
            # TODO: implement with LorentzFC?
            raise ValueError(f"Unknown value_agg mode: {self.value_agg}")
            
        if self.heads_concat:
            if self.concat_operation == "direct":
                out = self._concat_heads_direct(out)
            elif self.concat_operation == "log-radius":
                out = self._concat_heads_log_radius(out)
            else:
                raise NotImplementedError(f"Unknown concat_operation: {self.concat_operation}")
        else:
            #TODO: hyperbolic mean over heads?
            raise NotImplementedError("Only heads_concat is implemented")
        
        out = self.W_o(out)

        return out