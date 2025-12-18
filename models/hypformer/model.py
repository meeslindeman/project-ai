import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hypformer.lorentz import Lorentz
from models.hypformer.attention import HypformerAttention
from models.hypformer.decoder import HyperbolicCLS, HyperbolicLinear

class GraphClassifier(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int, 
        num_classes: int, 
        num_layers: int = 2,
        att_type: str = 'full', 
        decoder_type: str = 'linear', 
        num_heads: int = 1,
        k: float = 1.0,
        attn_mask: torch.Tensor = None
    ) -> None:
        super().__init__()
        self.attn_mask = attn_mask
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.manifold = Lorentz(k)

        # each attention layer maps hidden_dim -> hidden_dim (spatial),
        # producing Lorentz vectors of dim (hidden_dim + 1).
        self.attn_layers = nn.ModuleList([
            HypformerAttention(
                manifold=self.manifold,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                att_type=att_type,
                num_heads=num_heads,
                use_weight=True,
                heads_concat=False,
            )
            for _ in range(num_layers)
        ])

        if decoder_type == 'cls':
            self.decoder = HyperbolicCLS(
                manifold=self.manifold,
                input_dim=hidden_dim + 1,  
                num_classes=num_classes,
                bias=True
            )
        elif decoder_type == 'linear':
            self.decoder = HyperbolicLinear(
                manifold=self.manifold,
                input_dim=hidden_dim + 1, 
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin_in(x) 
        x = F.normalize(x, p=2, dim=-1) * 0.1

        for attn in self.attn_layers:
            x_lorentz = attn(x, attn_mask=self.attn_mask)
            x = x_lorentz[..., 1:]  # drop time-like component for next layer input as 'euc' is hardcoded

        logits = self.decoder(x_lorentz)             

        return logits
        

