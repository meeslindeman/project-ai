import torch
import torch.nn as nn
import torch.nn.functional as F
from manifolds.personal import LorentzMLR, Lorentz
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
            value_agg: str = "midpoint", 
            concat_operation: str = "direct", 
            a_default: float = 0.0,
            split_qkv: bool = False,
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
            split_qkv=split_qkv,
            debug=attn_debug,
        )

        self.attention2 = LorentzAttention(
            input_dim=embed_dim + 1,
            curvature=curvature_k,
            num_heads=num_heads,
            compute_scores=compute_scores,
            value_agg=value_agg,
            concat_operation=concat_operation,
            out_dim=embed_dim,   
            a_default=a_default,
            split_qkv=split_qkv,
            debug=attn_debug
        )

        self.manifold = Lorentz(curvature_k)

        self.fc = LorentzMLR(
            in_features=embed_dim,
            out_features=num_classes,
            k=curvature_k,
            reset_params="kaiming",
            input="lorentz"
        )

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # euclidean embeddings: [B, N, embed_dim]
        embeds = self.embedding(token_ids)
        embeds = F.normalize(embeds, p=2, dim=-1) * 0.1

        # map embeddings to Lorentz manifold: [B, N, 1 + embed_dim]
        x_lorentz = self.manifold.expmap0(embeds)

        # hyperbolic attention on the manifold: [B, N, 1 + embed_dim] (after heads + W_o)
        attn_output = self.attention(x_lorentz, mask=mask)

        attn_output = self.attention2(attn_output, mask=mask)

        # hyperbolic mean pooling over tokens: [B, 1 + embed_dim]
        pooled = self.manifold.pooling(attn_output, mask)   #TODO: frechet mean

        # Lorentz MLR classifier; expects Lorentz input: [B, num_classes]
        logits = self.fc(pooled)   

        return logits