import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma

from manifolds.personal import Lorentz, LorentzFC


class LorentzAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 1, heads_concat: bool = False, compute_scores: str = "linner", value_agg: str = "hexformer", learnable_temperature: bool = True, curvature: float = -1.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.heads_concat = heads_concat
        self.compute_scores = compute_scores
        self.value_agg = value_agg
        self.curvature = curvature       

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.head_dim = input_dim // num_heads

        # temperature (for Hexformer-style scaling)
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("temperature", torch.tensor(1.0))

        self.manifold = Lorentz()         
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_k = nn.ModuleList()
        self.W_q = nn.ModuleList()
        self.W_v = nn.ModuleList()
        for _ in range(self.num_heads):
            self.W_k.append(LorentzFC(self.input_dim, self.head_dim, activation=nn.Identity()))
            self.W_q.append(LorentzFC(self.input_dim, self.head_dim, activation=nn.Identity()))
            self.W_v.append(LorentzFC(self.input_dim, self.head_dim, activation=nn.Identity()))

        # Log-radius concat parameters (for heads_concat=True)
        # Per-head Lorentz dimension = head_dim = 1 + d_space_head
        d_space_head = self.head_dim - 1
        n_i = d_space_head                     # per-head spatial dimension
        n_total = num_heads * d_space_head     # total spatial dimension after concat

        # s(n, n_i) = exp(0.5 * (psi(n/2) - psi(n_i/2)))
        if d_space_head > 0:
            s_val = math.exp(
                0.5
                * (
                    digamma(torch.tensor(n_total / 2.0)).item()
                    - digamma(torch.tensor(n_i / 2.0)).item()
                )
            )
        else:
            # degenerate case: no spatial dims
            s_val = 1.0
        self.register_buffer("log_radius_scale", torch.tensor(float(s_val)))

    
    @staticmethod
    def _lorentz_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_t, x_x = x[..., :1], x[..., 1:]
        y_t, y_x = y[..., :1], y[..., 1:]

        time = torch.matmul(x_t, y_t.transpose(-1, -2))    # [B, H, N, N]
        space = torch.matmul(x_x, y_x.transpose(-1, -2))   # [B, H, N, N]

        # Minkowski metric: -t_x * t_y + <x_x, y_x>
        return -time + space


    @staticmethod
    def _lorentz_sqdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lp = self._lorentz_inner(x, y)          # [B, H, N, N]
        k = -self.curvature                     # k > 0 if curvature < 0
        d2 = -2.0 * k - 2.0 * lp
        return d2


    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        assert D == self.input_dim

        q_heads, k_heads, v_heads = [], [], []
        for h in range(self.num_heads):
            q_heads.append(self.W_q[h](x))  # [B, N, head_dim]
            k_heads.append(self.W_k[h](x))  # [B, N, head_dim]
            v_heads.append(self.W_v[h](x))  # [B, N, head_dim]

        # Stack into [B, H, N, head_dim]
        q = torch.stack(q_heads, dim=1)
        k = torch.stack(k_heads, dim=1)
        v = torch.stack(v_heads, dim=1)

        if self.compute_scores == "linner":
            # Plain Lorentz inner product scores
            scores = self._lorentz_inner(q, k)           # [B, H, N, N]
            scores = scores * self.scale
        elif self.compute_scores == "hexformer":
            # Hexformer-style scores: scaled negative squared distance
            d2 = self._lorentz_sqdist(q, k)              # [B, H, N, N]
            scores = -d2 * self.scale * self.temperature
        else:
            raise ValueError(f"Unknown compute_scores mode: {self.compute_scores}")

        if mask is not None:
            # mask: [B, N] -> [B, 1, 1, N]
            mask_exp = mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~mask_exp, float("-inf"))

        attn = F.softmax(scores, dim=-1)    # [B, H, N, N]

        if self.value_agg == "hexformer":
            # Hexformer-style: log/exp in tangent space at origin
            #   1) v_tan = log_0(v)     in tangent space
            #   2) out_tan = attn @ v_tan
            #   3) out = exp_0(out_tan) back on hyperboloid
            v_tan = self.manifold.logmap0(v)           # [B, H, N, D]
            out_tan = torch.matmul(attn, v_tan)        # [B, H, N, D]
            out = self.manifold.expmap0(out_tan)       # [B, H, N, D]
        elif self.value_agg == "midpoint":
            # Hypformer-style hyperbolic midpoint aggregation.
            # We want, for each (B, H, query_index), a weighted midpoint
            # of the N values along the key dimension.
            #
            # mid_point expects:
            #   x: (..., N_points, D)
            #   w: (..., 1, N_points)
            # and reduces over dim=-2.
            #
            # v:    [B, H, N_k, D]
            # attn: [B, H, N_q, N_k]  (usually N_q = N_k = N)
            B_, H_, N_k, D_ = v.shape
            N_q = attn.shape[2]

            # Broadcast values over query dimension:
            #   v_expanded: [B, H, N_q, N_k, D]
            v_expanded = v.unsqueeze(2).expand(B_, H_, N_q, N_k, D_)

            # Expand attention weights with singleton "point-axis":
            #   w_expanded: [B, H, N_q, 1, N_k]
            w_expanded = attn.unsqueeze(-2)

            # mid_point returns [B, H, N_q, 1, D]; then squeeze
            out = self.manifold.mid_point(v_expanded, w=w_expanded)  # [B, H, N_q, 1, D]
            out = out.squeeze(-2)                                    # [B, H, N_q, D]
        else:
            raise ValueError(f"Unknown value_agg mode: {self.value_agg}")


        if self.heads_concat:
            # Log-radius concatenation across heads (Intrinsic)
            # out: [B, H, N, D] with D = 1 + d_space
            B_, H_, N_, D_ = out.shape
            assert H_ == self.num_heads
            d_space = D_ - 1
            assert d_space > 0, "Need spatial dimensions for log-radius concat"

            # Time and spatial parts
            t_h = out[..., 0]          # [B, H, N]
            u_h = out[..., 1:]         # [B, H, N, d_space]

            s = self.log_radius_scale.to(out.device)    # scalar log-radius scale
            invK = 1.0 / self.curvature                 # 1/K; K < 0

            # Scale spatial parts for log-radius alignment
            u_tilde_h = s * u_h                          # [B, H, N, d_space]

            # Per-head Lorentz constraint: -t_i^2 + ||u_i||^2 = 1/K
            # => ||u_i||^2 = t_i^2 + 1/K
            # After scaling and concat:
            #   -t'^2 + s^2 * Sum_i (t_i^2 + 1/K) = 1/K
            # => t'^2 = s^2 * Sum_i (t_i^2 + 1/K) - 1/K
            sum_term = (t_h**2 + invK).sum(dim=1)       # [B, N]
            t_sq = (s**2) * sum_term - invK             # [B, N]
            t_sq = torch.clamp(t_sq, min=1e-8)
            t_prime = torch.sqrt(t_sq)                  # [B, N], upper sheet

            # Concatenate scaled spatial parts across heads
            u_tilde_perm = u_tilde_h.permute(0, 2, 1, 3)   # [B, N, H, d_space]
            u_concat = u_tilde_perm.reshape(B_, N_, H_ * d_space)  # [B, N, H*d_space]

            # Final concatenated Lorentz vector per token: [B, N, 1 + H*d_space]
            out = torch.cat([t_prime.unsqueeze(-1), u_concat], dim=-1)

        else:
            # TODO: hyperbolic mean pooling of heads instead of concatenation.
            raise NotImplementedError("Hyperbolic mean pooling over heads not implemented yet.")

        # #TODO: final projection / normalization?

        return out
