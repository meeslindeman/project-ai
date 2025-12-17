import torch
import torch.nn as nn
from models.hypformer.layer import HypCLS, HypLinear

class HyperbolicCLS(nn.Module):
    """Fully hyperbolic classifier head using HypCLS prototypes."""
    def __init__(self, manifold, input_dim: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.head = HypCLS(manifold=manifold, in_channels=input_dim-1, out_channels=num_classes, bias=bias)

    def forward(self, x_hyp: torch.Tensor) -> torch.Tensor:
        return self.head(x_hyp, x_manifold='hyp', return_type='neg_dist')


class HyperbolicLinear(nn.Module):
    """Hybrid decoder: logmap to Euclidean, drop time, linear classifier."""
    def __init__(self, manifold, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = HypLinear(manifold=manifold, in_features=input_dim-1 , out_features=num_classes)

    def forward(self, x_hyp: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x_hyp, x_manifold='hyp')
        return logits[..., 1:]
