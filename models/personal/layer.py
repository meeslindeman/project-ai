import os
import sys
import torch
import torch.nn as nn
from models.personal.lorentz import Lorentz

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(
    _here, "..", "..", "external", "hyperbolic-fully-connected"
)
_repo_root = os.path.normpath(_repo_root)
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from layers.LLinear import LorentzFullyConnected

LorentzFC = LorentzFullyConnected

class LorentzMLR(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: float = 0.1, reset_params: str = "kaiming", a_default: float = 0.0, activation: nn.Module = nn.Identity(), input_space: str = "lorentz") -> None:
        super().__init__()
        self.manifold = Lorentz(k)
        self.linear = LorentzFC(in_features, out_features, self.manifold, reset_params=reset_params, a_default=a_default, activation=activation)
        self.input_space = input_space
    
    def forward(self, x):
        if self.input_space == "euclidean":
            x = self.manifold.expmap0(x)
        return self.linear.compute_output_space(x) 