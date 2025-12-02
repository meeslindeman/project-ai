# TODO: Credit Ricardo

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

# ======= Temp fixes =========
_EPS = {
    torch.float32: 1e-7,
    torch.float64: 1e-15,
}

# from lorentz_math
def arcosh(x: torch.Tensor) -> torch.Tensor:
    """Computes the inverse hyperbolic cosine safely."""
    dtype = x.dtype
    z = torch.sqrt(torch.clamp_min(x.double().pow(2) - 1.0, _EPS[torch.float64]))
    return torch.log(x.double() + z).to(dtype)
# ===========================

class Lorentz(nn.Module):
    """Lorentz model for hyperbolic geometry.
    The Lorentz model is defined as the hyperboloid in Minkowski spacel
    The manifold is defined by the equation:
        -x+0^2 + x_1^2 + ... + x_n^2 = -1/k"""
    def __init__(self, k: float = 0.1, requires_grad: bool = False, contraining_strategy: nn.Module = nn.Identity()) -> None:
        super().__init__()
        k_value = torch.log(torch.exp(torch.tensor(k)) - 1)
        self.c_softplus_inv = nn.Parameter(k_value, requires_grad=requires_grad)
        self.constraining_strategy = contraining_strategy
    
    def k(self):
        """Returns the negative curvature of the Lorentz model."""
        return F.softplus(self.c_softplus_inv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expmap0(x)

    def normL(self, x: torch.Tensor, metric: bool = None) -> torch.Tensor:
        if metric is None:
            metric = torch.ones(x.size(-1), device=x.device, dtype=x.dtype)
        metric[0] = -1
        return (x * x * metric).sum(dim=-1, keepdim=True).sqrt()
    
    def expmap0(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps tangent vectors from the origin of the tangent space T_0 H^n_k
        to the Lorentz hyperboloid H^n_K.
        Handles the case where the input vector norm is zero.
        """
        sqrt_k = self.k().sqrt()
        norm_x = torch.norm(x, dim=-1, keepdim=True)

        eps = 1e-9
        is_zero = norm_x < eps

        sqrt_k_norm_x = sqrt_k * norm_x
        time = 1.0 / sqrt_k * torch.cosh(sqrt_k_norm_x)

        factor = torch.where(
            is_zero,
            torch.ones_like(norm_x),
            torch.sinh(sqrt_k_norm_x) / (sqrt_k_norm_x)
        )

        space = factor * x
        return torch.cat([time, space], dim=-1)
    
    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        """
        Maps points from the Lorentz hyperboloid H^n_k
        back to the tangent space T_0 H^n_k at the origin.
        """
        k = self.k()
        sqrt_k = k.sqrt()
        # update
        dtype = y.dtype
        eps = _EPS.get(dtype, 1e-7)
        
        y_time = y[..., :1] # First component (time)
        y_space = y[..., 1:] # Remaining components (space)

        # eps = 1e-9

        # Calculate the factor based on the formula
        # arccosh(sqrt(k) * y_time) / sqrt((sqrt(k) * y_time)^2 - 1)
        # The argument to sqrt can be negative due to floating point errors, so clamp at 0.
        norm_y_space_sq = torch.sum(y_space * y_space, dim=-1, keepdim=True)
        denom_sqrt = torch.sqrt(torch.clamp(k * norm_y_space_sq, min=eps))

        acosh_arg = (sqrt_k * y_time).clamp_min(1.0 + eps)
        factor = arcosh(acosh_arg) / denom_sqrt
        # factor = torch.acosh(sqrt_k * y_time) / denom_sqrt

        return factor * y_space
    
    def projection_space_orthogonal(self, x: torch.Tensor) -> torch.Tensor:
        """Projects a point onto the Lorentz model orthogonally from the space dimensions."""
        return torch.cat([torch.sqrt(1/self.k() + x.pow(2).sum(dim=-1, keepdim=True)), x], dim=-1)
    
    def poincare_to_lorentz(self, x: torch.Tensor, x_poincare: torch.Tensor) -> torch.Tensor:
        """
        Converts points from the Poincaré ball model to the Lorentz hyperboloid model.
        The conversion assumes both models share the same curvature parameter k > 0.
        """
        k = self.k()
        sqrt_k = k.sqrt()

        # Calculate the squared Euclidean norm of the Poincaré points
        x_norm_sq = torch.sum(x_poincare * x_poincare, dim=-1, keepdim=True)
        
        # Denominator for the conversion formula
        # Add epsilon for numerical stability
        denom = 1 - k * x_norm_sq + 1e-9
        
        # Time component of the Lorentz point
        time_component = (1 / sqrt_k) * (1 + k * x_norm_sq) / denom
        
        # Space components of the Lorentz point
        space_components = (2 * x_poincare) / denom
        
        # Concatenate time and space to form the Lorentz point
        return torch.cat([time_component, space_components], dim=-1)

    def inner(self, u: torch.Tensor, v: Optional[torch.Tensor] = None, *, keepdim=False, dim=-1) -> torch.Tensor:
        if v is None:
            v = u

        d = u.size(dim) - 1
        uv = u * v
        if not keepdim:
            return -uv.narrow(dim, 0, 1).squeeze(dim) + uv.narrow(dim, 1, d).sum(dim=dim, keepdim=False)
        else:
            return -uv.narrow(dim, 0, 1) + uv.narrow(dim, 1, d).sum(dim=dim, keepdim=True)

    def mid_point(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, *, keepdim=False, dim=-1) -> torch.Tensor:
        # 1) Minkowski linear combination over token dimension (-2)
        if w is not None:
            # Ensure weights are normalized over the key dimension
            # (safe even if they already come from softmax).
            w_norm = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-8))

            # w: [..., M, N], x: [..., N, D] -> ave: [..., M, D]
            ave = w_norm.matmul(x)
        else:
            ave = x.mean(dim=-2)  # [..., D]

        # 2) Project back to the hyperboloid <y,y>_L = -1/k
        k = self.k()

        # Minkowski norm squared s = <ave, ave>
        s = self.inner(ave, ave, keepdim=True, dim=dim)  # [..., 1]

        # We want alpha^2 * s = -1/k  => alpha^2 = -1/(k*s)
        # Clamp s away from 0 to avoid blow-ups.
        s_clamped = torch.clamp(s, max=-1e-8)  # ensure s <= -1e-8

        alpha_sq = -1.0 / (k * s_clamped)      # [..., 1]
        alpha_sq = alpha_sq.clamp_min(1e-8)
        alpha = alpha_sq.sqrt()                # [..., 1]

        y = ave * alpha  # [..., D]

        if keepdim:
            # If you need a token-like dim preserved, you can unsqueeze here.
            # For current use (attention & pooling), keepdim=False is usually fine.
            y = y.unsqueeze(-2)

        return y