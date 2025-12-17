import torch
import torch.nn as nn
import torch.nn.functional as F
from manifolds.hypformer import Lorentz
from models.hypformer.attention import HypformerAttention
from models.hypformer.decoder import HyperbolicCLS, HyperbolicLinear

class Classifier(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embed_dim: int, num_classes: int, att_type: str = 'full', decoder_type: str = 'linear', num_heads: int = 1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        self.manifold = Lorentz(k=1.0)
        self.attention = HypformerAttention(manifold=self.manifold, input_dim=embed_dim, att_type=att_type, num_heads=num_heads, use_weight=True, heads_concat=False)
        
        if decoder_type == 'cls':
            self.decoder = HyperbolicCLS(
                manifold=self.manifold,
                input_dim=embed_dim + 1,  
                num_classes=num_classes,
                bias=True
            )
        elif decoder_type == 'linear':
            self.decoder = HyperbolicLinear(
                manifold=self.manifold,
                input_dim=embed_dim + 1, 
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        
    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(token_ids)
        embeds = F.normalize(embeds, p=2, dim=-1) * 0.1 # Needed for exploding values
        att_output = self.attention(embeds, mask)
        pooled = self.manifold.mid_point(att_output)
        logits = self.decoder(pooled)
        return logits


