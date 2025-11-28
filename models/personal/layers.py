import torch
import torch.nn as nn

from manifolds.personal import Lorentz

class EuclidLinear(nn.Module):
    def __init__(self, in_lorentz_dim: int, out_spatial_dim: int, manifold: Lorentz):
        super().__init__()
        self.manifold = manifold
        self.in_lorentz_dim = in_lorentz_dim
        self.out_spatial_dim = out_spatial_dim

        self.linear = nn.Linear(in_lorentz_dim - 1, out_spatial_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        space = x[..., 1:]               
        y_space = self.linear(space)      
        y_lorentz = self.manifold.projection_space_orthogonal(y_space)  
        return y_lorentz
