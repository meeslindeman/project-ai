import torch
import torch.nn as nn

import numpy

from manifolds.personal.layer import LorentzFC
from manifolds.personal.lorentz import Lorentz

def _minkowski_norm_sq(x: torch.Tensor) -> torch.Tensor:
    time = x[..., :1]   
    space = x[..., 1:] 
    norm_sq = -time * time + torch.sum(space * space, dim=-1, keepdim=True)
    return norm_sq

def _is_on_manifold(x: torch.Tensor, manifold: Lorentz, tol: float = 1e-4, log_details: bool = False) -> bool:
    k_val = manifold.k().item()
    target = -1.0 / k_val

    norm_sq = _minkowski_norm_sq(x)
    diff = norm_sq - target
    max_diff = diff.abs().max().item()

    if log_details:
        logger.debug("[manifold check] k=%.6f", k_val)
        logger.debug("  target Minkowski norm = %.6f", target)
        logger.debug("  max |<x,x>_L - target| = %.6e", max_diff)

    return max_diff < tol

matrix = torch.rand(3, 3)
vector = torch.rand(3)

result = torch.matmul(matrix, vector)

in_feats = 3
out_feats = 3

linear = nn.Linear(in_feats, out_feats, bias=False)
linear.weight.data = matrix

layer_result = linear(vector)

manifold = Lorentz(0.1)

lorentz_fc = LorentzFC(
    in_features=3,
    out_features=3,
    manifold=manifold,
    reset_params="kaiming",
    a_default=0.0,
    activation=nn.Identity()
)

with torch.no_grad():
    lorentz_fc.U.copy_(matrix)
    lorentz_fc.a.zero_()

x = torch.cat([vector, torch.zeros(1)], dim=0).unsqueeze(0)
layer_out = lorentz_fc.compute_output_space(x).squeeze(0) 

print("Manual result: ", result)
print("Layer result: ", layer_result)
print("LorentzFC compute_output_space: ", layer_out)
print("Difference:", (layer_out - result).abs().max().item())