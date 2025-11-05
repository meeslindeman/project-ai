import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from manifolds.hypformer import Lorentz
from manifolds.hypformer.layer import HypLinear

class HypformerAttention(nn.Module):
    def __init__(self, manifold: Lorentz, input_dim: int, att_type: str = "full", num_heads: int = 1, use_weight: bool = True, heads_concat: bool = False) -> None:
        super().__init__()
        self.manifold = manifold
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.att_type = att_type
        self.use_weight = use_weight
        self.heads_concat = heads_concat

        assert input_dim % num_heads == 0, \
            f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = input_dim // num_heads

        self.Wk = nn.ModuleList()
        self.Wq = nn.ModuleList()
        for i in range(self.num_heads):
            self.Wk.append(HypLinear(self.manifold, self.input_dim, self.head_dim))
            self.Wq.append(HypLinear(self.manifold, self.input_dim, self.head_dim))
        
        if use_weight:
            self.Wv = nn.ModuleList()
            for i in range(self.num_heads):
                self.Wv.append(HypLinear(self.manifold, self.input_dim, self.head_dim))
        
        if self.att_type == 'full':
            self.scale = nn.Parameter(torch.tensor([math.sqrt(self.head_dim)]))
            self.bias = nn.Parameter(torch.zeros(()))
        elif self.att_type == 'linear_focused':
            self.norm_scale = nn.Parameter(torch.ones(()))
            self.power_k = 2.0
            self.v_map_mlp = nn.Linear(self.head_dim, self.head_dim, bias=True)
        
        if heads_concat:
            raise NotImplementedError(
                "heads_concat=True not implemented in original code "
                "(references undefined self.final_linear)"
            )
        
        if num_heads > 1:
            raise NotImplementedError(
                "Multi-head (num_heads > 1) creates dimension mismatch. "
                "Use num_heads=1 for now. Multi-head support is future work."
            )

    
    @staticmethod
    def fp(x: torch.Tensor, p: int = 2) -> torch.Tensor:
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p
    
    def full_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, output_attn: bool = False) -> torch.Tensor:
        B, N, H, D = query.shape                                    # [B, N, H, D]

        query = query.transpose(1, 2)                               # [B, H, N, D]
        key = key.transpose(1, 2)                                   # [B, H, N, D]
        value = value.transpose(1, 2)                               # [B, H, N, D]

        att_weight = 2.0 + 2.0 * self.manifold.cinner(query, key)   # [B, H, N, N]
        att_weight = att_weight / self.scale + self.bias            # [B, H, N, N]

        if mask is not None:
            att_weight = att_weight.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        att_weight = F.softmax(att_weight, dim=-1)                  # [B, H, N, N]
        att_output = self.manifold.mid_point(value, att_weight)     # [B, H, N, D]
         
        att_output = att_output.transpose(1, 2)                     # [B, N, H, D]

        if self.num_heads == 1:
            att_output = att_output.squeeze(2)                      # [B, N, D]
        else:
            # Hyperbolic mean over heads (dim=2)
            att_output = self.manifold.mid_point(att_output)        # [B, N, D]
        
        if output_attn:
            return att_output, att_weight
        return att_output

    def linear_focus_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, output_attn: bool = False) -> torch.Tensor:
        B, N, H, D = query.shape

        query = query.transpose(1, 2)                               # [B, H, N, D]
        key = key.transpose(1, 2)                                   # [B, H, N, D]
        value = value.transpose(1, 2)                               # [B, H, N, D]

        # Remove time dimension
        qs = query[..., 1:]                                         # [B, H, N, D-1]
        ks = key[..., 1:]                                           # [B, H, N, D-1]
        vs = value[..., 1:]                                         # [B, H, N, D-1]

        eps = 1e-6
        phi_qs = (F.relu(qs) + eps) / (self.norm_scale.abs() + eps)  
        phi_ks = (F.relu(ks) + eps) / (self.norm_scale.abs() + eps)  

        phi_qs = self.fp(phi_qs, p=self.power_k)    # [B, H, N, D-1]
        phi_ks = self.fp(phi_ks, p=self.power_k)    # [B, H, N, D-1]

        # K^T V: sum over N dimension
        k_transpose_v = torch.einsum('bhnm,bhnd->bhmd', phi_ks, vs)  # [B, H, D-1, D-1]

        # Q(K^T V)
        numerator = torch.einsum('bhnm,bhmd->bhnd', phi_qs, k_transpose_v)  # [B, H, N, D-1]

        # Sum of kernel-transformed keys
        sum_ks = torch.einsum('bhnd->bhd', phi_ks)  # [B, H, D-1]
        denominator = torch.einsum('bhnd,bhd->bhn', phi_qs, sum_ks)  # [B, H, N]
        denominator = denominator.unsqueeze(-1)  # [B, H, N, 1]

        # Normalize
        attn_output = numerator / (denominator + eps)  # [B, H, N, D-1]

        # Residual
        vss = self.v_map_mlp(vs)  # [B, H, N, D-1]
        attn_output = attn_output + vss

        attn_output = attn_output.transpose(1, 2)                   # [B, N, H, D]

        # Aggregate heads
        if self.num_heads == 1:
            attn_output = attn_output.squeeze(2)                    # [B, N, D-1]

        #TODO: trans_head_concat
        
        else:
            # Mean over heads (Euclidean mean in space dims)
            attn_output = attn_output.mean(dim=2)                   # [B, N, D-1]

        attn_output_time = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k) ** 0.5
        attn_output = torch.cat([attn_output_time, attn_output], dim=-1)  # [B, H, N, D]

        if output_attn:
            return attn_output, attn_output
        return attn_output

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q_list = []
        k_list = []
        v_list = []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](x, x_manifold='euc'))
            k_list.append(self.Wk[i](x, x_manifold='euc'))
            if self.use_weight:
                v_list.append(self.Wv[i](x, x_manifold='euc'))
            else:
                raise NotImplementedError("Not yet done")
        
        query = torch.stack(q_list, dim=2)  # [B, N, H, D]
        key = torch.stack(k_list, dim=2)    # [B, N, H, D]
        value = torch.stack(v_list, dim=2)  # [B, N, H, D]

        if self.att_type == 'full':
            output = self.full_attention(query, key, value, mask)
        elif self.att_type == 'linear_focused':
            output = self.linear_focus_attention(query, key, value, mask)

        return output