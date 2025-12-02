import torch
import torch.nn as nn
import torch.nn.functional as F
from manifolds.personal import LorentzMLR, Hyperboloid
from models.personal.attention import LorentzAttention

class Classifier(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            pad_id: int, 
            embed_dim: int, 
            num_classes: int, 
            curvature_k: float = 0.1, 
            num_heads: int = 1, 
            compute_scores: str = "lorentz_inner", 
            value_agg: str = "riemannian", 
            concat_operation: str = "direct", 
            a_default: float = 0.0,
            attn_debug: bool = False
        ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.curvature_k = curvature_k

        #TODO: init poincare and map to Lorentz?
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id) 
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        
        self.attention = LorentzAttention(
            input_dim=embed_dim + 1,
            curvature=curvature_k,
            num_heads=num_heads,
            compute_scores=compute_scores,
            value_agg=value_agg,
            concat_operation=concat_operation,
            out_dim=embed_dim,   
            a_default=a_default,
            debug=attn_debug,
        )
        self.manifold = self.attention.manifold
        
        #TODO: currently taking gyro ops from Chen; add operations to own manifold
        self.hyperboloid = Hyperboloid() 

        self.fc = LorentzMLR(
            in_features=embed_dim,
            out_features=num_classes,
            k=curvature_k,
            reset_params="kaiming",
            input="lorentz"
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
                # if an entire row is masked: just take first token
                pooled_b = x[b:b+1, 0:1, :]   # [1, D]
            elif xb.size(0) == 1:
                pooled_b = xb[0:1]            # [1, D]
            else:
                # Mobius add on hyperboloid
                acc = xb[0:1]                 # [1, D]
                for i in range(1, xb.size(0)):
                    acc = hyp.mobius_add(acc, xb[i:i+1], c)  # [1, D]

                # divide by n via scalar Mobius multiplication
                pooled_b = hyp.mobius_scalarmul(1.0 / xb.size(0), acc, c)  # [1, D]

            pooled_list.append(pooled_b)

        pooled = torch.cat(pooled_list, dim=0)  # [B, D_lorentz]
        return pooled
    
    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # euclidean embeddings: [B, N, embed_dim]
        embeds = self.embedding(token_ids)
        embeds = F.normalize(embeds, p=2, dim=-1) * 0.1

        # map embeddings to Lorentz manifold: [B, N, 1 + embed_dim]
        x_lorentz = self.manifold.expmap0(embeds)

        # hyperbolic attention on the manifold: [B, N, 1 + embed_dim] (after heads + W_o)
        attn_output = self.attention(x_lorentz, mask=mask)

        # hyperbolic mean pooling over tokens: [B, 1 + embed_dim]
        pooled = self.pooling(attn_output, mask)  

        # Lorentz MLR classifier; expects Lorentz input: [B, num_classes]
        logits = self.fc(pooled)   

        return logits