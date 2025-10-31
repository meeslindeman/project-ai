import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hypformer.layer import HypLinear
from geoopt import Manifold

class HypAttention(nn.Module):
    def __init__(self, manifold: Manifold, in_channels: int, out_channels: int, num_heads: int,
                 use_weight: bool = True, attention_type: str = "full",
                 power_k: int = 2, trans_heads_concat: bool = False) -> None:
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
        self.attention_type = attention_type
        self.power_k = power_k
        self.trans_heads_concat = trans_heads_concat

        self.Wk = nn.ModuleList([HypLinear(self.manifold, in_channels, out_channels) for _ in range(num_heads)])
        self.Wq = nn.ModuleList([HypLinear(self.manifold, in_channels, out_channels) for _ in range(num_heads)])
        if use_weight:
            self.Wv = nn.ModuleList([HypLinear(self.manifold, in_channels, out_channels) for _ in range(num_heads)])

        self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
        self.bias = nn.Parameter(torch.zeros(()))
        self.norm_scale = nn.Parameter(torch.ones(()))
        self.v_map_mlp = nn.Linear(out_channels, out_channels, bias=True)

        if trans_heads_concat:
            self.final_linear = nn.Linear(num_heads * out_channels, out_channels, bias=True)

    @staticmethod
    def fp(x: torch.Tensor, p: float = 2.0) -> torch.Tensor:
        eps = 1e-12
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / (norm_x_p + eps)) * (x ** p)

    def full_attention(self, qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, output_attn: bool = False) -> torch.Tensor:
        att_weight = 2 + 2 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))   # [H, N, N]
        att_weight = att_weight / self.scale + self.bias
        att_weight = nn.Softmax(dim=-1)(att_weight)
        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight).transpose(0, 1) # [N, H, D+1]
        att_output = self.manifold.mid_point(att_output)                                      # [N, D+1]
        return (att_output, att_weight) if output_attn else att_output

    def linear_focus_attention(self, hyp_qs: torch.Tensor, hyp_ks: torch.Tensor, hyp_vs: torch.Tensor, output_attn: bool = False) -> torch.Tensor:
        qs = hyp_qs[..., 1:]; ks = hyp_ks[..., 1:]; v = hyp_vs[..., 1:]  # [N, H, D], D=out_channels
        phi_qs = (F.relu(qs) + 1e-6) / (self.norm_scale.abs() + 1e-6)
        phi_ks = (F.relu(ks) + 1e-6) / (self.norm_scale.abs() + 1e-6)
        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        kT_v = torch.einsum('nhm,nhd->hmd', phi_ks, v)                      # [H, D, D]
        numerator = torch.einsum('nhm,hmd->nhd', phi_qs, kT_v)              # [N, H, D]
        sum_k = torch.einsum('nhd->hd', phi_ks)                              # [H, D]
        denominator = torch.einsum('nhd,hd->nh', phi_qs, sum_k).unsqueeze(-1)
        attn_output = numerator / (denominator + 1e-6)

        attn_output = attn_output + self.v_map_mlp(v)                        # [N, H, D]

        if self.trans_heads_concat:
            attn_output = self.final_linear(attn_output.reshape(attn_output.size(0), -1))  # [N, D]
        else:
            attn_output = attn_output.mean(dim=1)                                          # [N, D]

        t = ((attn_output ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        attn_output = torch.cat([t, attn_output], dim=-1)                                   # [N, D+1]
        return (attn_output, attn_output) if output_attn else attn_output

    def forward(self, query_input: torch.Tensor, source_input: torch.Tensor = None, output_attn: bool = False):
        if source_input is None:
            source_input = query_input

        q_list, k_list, v_list = [], [], []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](query_input))
            k_list.append(self.Wk[i](source_input))
            v_list.append(self.Wv[i](source_input) if self.use_weight else source_input)

        query = torch.stack(q_list, dim=1)  # [N, H, D+1]
        key   = torch.stack(k_list, dim=1)  # [N, H, D+1]
        value = torch.stack(v_list, dim=1)  # [N, H, D+1]

        if self.attention_type == 'linear_focused':
            return self.linear_focus_attention(query, key, value, output_attn)
        elif self.attention_type == 'full':
            return self.full_attention(query, key, value, output_attn)
        else:
            raise NotImplementedError
