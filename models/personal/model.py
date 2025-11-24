import torch
import torch.nn as nn
from manifolds.personal import LorentzMLR, Hyperboloid
from models.personal.attention import LorentzAttention


class Classifier(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embed_dim: int, num_classes: int, k: float = 0.1, num_heads: int = 1, 
                 compute_scores: str = "lorentz_inner", value_agg: str = "riemannian", concat_operation: str = "direct") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k

        # Euclidean embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = LorentzAttention(
            input_dim=embed_dim + 1,
            curvature=k,
            num_heads=num_heads,
            heads_concat=True,
            compute_scores=compute_scores,
            value_agg=value_agg,
            concat_operation=concat_operation,
            out_dim=embed_dim,    # spatial dim after attention
        )
        self.manifold = self.attention.manifold
        
        # Hyperboloid ops for pooling
        self.hyperboloid = Hyperboloid() #TODO: add operations to own manifold

        self.fc = LorentzMLR(
            in_features=embed_dim,
            out_features=num_classes,
            k=k,
            reset_params="kaiming",
            activation=nn.Identity(),
            input="lorentz",      # pooled is now Lorentz, not Euclidean
        )

    def pooling(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hyp = self.hyperboloid
        c = self.manifold.k().item()

        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)

        B, N, D = x.shape
        pooled_list = []

        for b in range(B):
            xb = x[b][mask[b]]  # [n_valid, D]

            if xb.size(0) == 0:
                # Fallback if an entire row is masked: just take first token
                pooled_b = x[b:b+1, 0:1, :]   # [1, D]
            elif xb.size(0) == 1:
                pooled_b = xb[0:1]            # [1, D]
            else:
                # Step 1: reduce via Möbius add on hyperboloid
                acc = xb[0:1]                 # [1, D]
                for i in range(1, xb.size(0)):
                    acc = hyp.mobius_add(acc, xb[i:i+1], c)  # [1, D], stays on manifold

                # Step 2: divide by n via scalar Möbius multiplication
                pooled_b = hyp.mobius_scalarmul(1.0 / xb.size(0), acc, c)  # [1, D]

            pooled_list.append(pooled_b)

        pooled = torch.cat(pooled_list, dim=0)  # [B, D_lorentz]
        return pooled
    
    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1) Euclidean embeddings: [B, N, embed_dim]
        embeds = self.embedding(token_ids)

        # 2) Map embeddings to Lorentz manifold: [B, N, 1 + embed_dim]
        x_lorentz = self.manifold.expmap0(embeds)

        # 3) Hyperbolic attention on the manifold
        #    Output: [B, N, 1 + embed_dim] (after heads + W_o)
        attn_output = self.attention(x_lorentz, mask=mask)

        # 4) Hyperbolic mean pooling over tokens (still Lorentz)
        pooled = self.pooling(attn_output, mask)  # [B, 1 + embed_dim]

        # 5) Final Lorentz MLR classifier; expects Lorentz input and outputs Euclidean logits
        logits = self.fc(pooled)                  # [B, num_classes]

        return logits