import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hypformer.lorentz import Lorentz
from models.hypformer.layer import HypLinear

class HypformerAttention(nn.Module):
    def __init__(
        self, 
        manifold: Lorentz, 
        in_channels: int, 
        out_channels: int,
        att_type: str = "full", 
        num_heads: int = 1, 
        use_weight: bool = True, 
        power_k: int = 2,
        heads_concat: bool = False,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.att_type = att_type
        self.use_weight = use_weight
        self.power_k = power_k

        spatial_in = in_channels - 1
        spatial_out = out_channels - 1

        # never actually implemented in orginal code
        self.heads_concat = heads_concat

        # each head has its own HypLinear(in -> out)
        self.Wk = nn.ModuleList([HypLinear(manifold, spatial_in, spatial_out) for _ in range(num_heads)])
        self.Wq = nn.ModuleList([HypLinear(manifold, spatial_in, spatial_out) for _ in range(num_heads)])

        if use_weight:
            self.Wv = nn.ModuleList([HypLinear(manifold, spatial_in, spatial_out) for _ in range(num_heads)])

        self.Wo = HypLinear(manifold, spatial_in, spatial_out)
        
        if self.att_type == 'full':
            self.scale = nn.Parameter(torch.tensor([math.sqrt(out_channels)]))
            self.bias = nn.Parameter(torch.zeros(()))
        elif self.att_type == 'linear_focused':
            self.norm_scale = nn.Parameter(torch.ones(()))
            self.v_map_mlp = nn.Linear(out_channels, out_channels, bias=True)

    @staticmethod
    def fp(x: torch.Tensor, p: int = 2) -> torch.Tensor:
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        norm_x_p = torch.norm(x ** p, p=2, dim=-1, keepdim=True)
        return (norm_x / norm_x_p) * x ** p
    
    def full_attention(self, qs: torch.Tensor, ks: torch.Tensor, vs: torch.Tensor, attn_mask: torch.Tensor = None, output_attn: bool = False):
        # qs,ks,vs: [N,H,D_lor], D_lor = out_channels+1 in Lorentz
        att_weight = 2.0 + 2.0 * self.manifold.cinner(qs.transpose(0, 1), ks.transpose(0, 1))  # [H,N,N]
        att_weight = att_weight / self.scale + self.bias

        if attn_mask is not None:
            # att_weight is [H, N, N]
            att_weight = att_weight.masked_fill(~attn_mask[None, :, :], -1e9)

        att_weight = F.softmax(att_weight, dim=-1)  # [H,N,N]

        att_output = self.manifold.mid_point(vs.transpose(0, 1), att_weight)  # typical [N,H,D_lor] (as in paper)
        if att_output.shape[0] == self.num_heads:  # if [H,N,D_lor]
            att_output = att_output.transpose(0, 1)  # -> [N,H,D_lor]

        # midpoint over heads
        att_output = self.manifold.mid_point(att_output)  # [N,D_lor]

        return (att_output, att_weight) if output_attn else att_output

    def linear_focus_attention(self, hyp_qs: torch.Tensor, hyp_ks: torch.Tensor, hyp_vs: torch.Tensor, output_attn: bool = False):
        qs = hyp_qs[..., 1:]  # [N,H,D]
        ks = hyp_ks[..., 1:]  # [N,H,D]
        vs = hyp_vs[..., 1:]  # [N,H,D]

        eps = 1e-6
        phi_qs = (F.relu(qs) + eps) / (self.norm_scale.abs() + eps)
        phi_ks = (F.relu(ks) + eps) / (self.norm_scale.abs() + eps)

        phi_qs = self.fp(phi_qs, p=self.power_k)
        phi_ks = self.fp(phi_ks, p=self.power_k)

        k_transpose_v = torch.einsum("nhm,nhd->hmd", phi_ks, vs)  # [H,D,D]
        numerator = torch.einsum("nhm,hmd->nhd", phi_qs, k_transpose_v)  # [N,H,D]

        sum_phi_k = torch.einsum("nhd->hd", phi_ks)  # [H,D]
        denominator = torch.einsum("nhd,hd->nh", phi_qs, sum_phi_k).unsqueeze(-1)  # [N,H,1]

        attn_output = numerator / (denominator + eps)  # [N,H,D]

        # v_map_mlp is applied to spatial v
        vss = self.v_map_mlp(vs)  # [N,H,D]
        attn_output = attn_output + vss  # [N,H,D]

        if self.heads_concat:
            attn_output = self.final_linear(attn_output.reshape(attn_output.size(0), -1))  # [N,D]
        else:
            attn_output = attn_output.mean(dim=1)  # [N,D]

        # recompute time component
        attn_time = ((attn_output ** 2).sum(dim=-1, keepdim=True) + self.manifold.k).sqrt()  # [N,1]
        attn_output = torch.cat([attn_time, attn_output], dim=-1)  # [N,D_lor]

        return (attn_output, attn_output) if output_attn else attn_output

    def forward(self, x_lorentz: torch.Tensor, attn_mask: torch.Tensor | None = None, output_attn: bool = False):
        # x_euc: [N,in_channels] in Euclidean space
        q_list, k_list, v_list = [], [], []
        for i in range(self.num_heads):
            q_list.append(self.Wq[i](x_lorentz, x_manifold='hyp'))   # manifold input
            k_list.append(self.Wk[i](x_lorentz, x_manifold='hyp'))
            if self.use_weight:
                v_list.append(self.Wv[i](x_lorentz, x_manifold='hyp'))
            else:
                v_list.append(x_lorentz)

        qs = torch.stack(q_list, dim=1)  # [N,H,D_lor]
        ks = torch.stack(k_list, dim=1)
        vs = torch.stack(v_list, dim=1) if self.use_weight else None

        if not self.use_weight:
            raise ValueError("use_weight=False is not supported with Euclidean x input; provide Lorentz source_input instead.")

        if self.att_type == "full":
            return self.full_attention(qs, ks, vs, attn_mask=attn_mask, output_attn=output_attn)
        elif self.att_type == "linear_focused":
            # adjacency-based attention not implemented for linear_focused
            return self.linear_focus_attention(qs, ks, vs, output_attn=output_attn)
        else:
            raise NotImplementedError(f"Unknown attention type: {self.att_type}")