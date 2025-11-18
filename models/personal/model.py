import torch
import torch.nn as nn
from manifolds.personal import LorentzMLR
from manifolds.personal import Hyperboloid

class Classifier(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, embed_dim: int, num_classes: int, **kwargs) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = Attention(embed_dim) # IMPLEMENT
        self.fc = LorentzMLR(embed_dim, num_classes)
        self.hyperboloid = Hyperboloid()
    
    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(token_ids)         
        attn_output = self.attention(embeds, mask)  
        pooled = attn_output.mean(dim=1) # IMPLEMENT
        logits = self.fc(pooled)                   
        return logits