import torch
import torch.nn as nn

from hypformer.attention import HypAttention
from hypformer.layer import HypCLS
from geoopt import Manifold

class HypClassifier(nn.Module):
    def __init__(
        self,
        manifold: Manifold,
        vocab_size: int,
        pad_id: int,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 4,
        attention_type: str = "full",           # "full" or "linear_focused"
        trans_heads_concat: bool = False,
        use_weight: bool = True,
        embed_scale: float = 0.1,               # scale before expmap0
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)

        self.attn = HypAttention(
            manifold=manifold,
            in_channels=embed_dim,
            out_channels=embed_dim,
            num_heads=num_heads,
            use_weight=use_weight,
            attention_type=attention_type,
            trans_heads_concat=trans_heads_concat,
        )

        self.cls = HypCLS(manifold, in_channels=embed_dim, out_channels=num_classes, bias=True)
        self.embed_scale = embed_scale

    def _to_lorentz(self, x_eucl: torch.Tensor) -> torch.Tensor:
        t0 = torch.zeros_like(x_eucl[..., :1])
        x_tan = torch.cat([t0, x_eucl * self.embed_scale], dim=-1)
        return self.manifold.expmap0(x_tan)  # [B,N,D+1]

    def _mask_weights(self, mask: torch.Tensor) -> torch.Tensor:
        w = mask.float()
        return w / w.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # embeddings -> Lorentz
        x_e = self.embed(token_ids)          # [B,N,D]
        x_h = self._to_lorentz(x_e)          # [B,N,D+1]

        # attention per batch (HypAttention expects sequence-first)
        B, N, _ = x_h.shape
        att_seq = []
        for b in range(B):
            att_out = self.attn(query_input=x_h[b, :N], source_input=None, output_attn=False)  # [N,D+1]
            att_seq.append(att_out)
        att_out = torch.stack(att_seq, dim=0)  # [B,N,D+1]

        # Lorentz pooling with mask
        w = self._mask_weights(mask)                               # [B,N]
        x_log = self.manifold.logmap0(att_out)                     # [B,N,D+1]
        mu_tan = (w.unsqueeze(-1) * x_log).sum(dim=1)              # [B,D+1]
        pooled = self.manifold.expmap0(mu_tan)                     # [B,D+1]

        # hyperbolic classifier logits
        logits = self.cls(pooled, x_manifold='hyp', return_type='neg_dist')  # [B,K]
        return logits
