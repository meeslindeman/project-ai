# TODO: Credit Ricardo

import torch
import torch.nn as nn
from lorentz import Lorentz

class LorentzFC(nn.Module):
    def __init__(self, in_features: int, out_features: int, manifold: Lorentz = Lorentz(0.1), reset_params: str = "eye", activation: nn.Module = nn.functional.relu) -> None:
        super().__init__()
        self.manifold = manifold
        self.U = nn.Parameter(torch.rand(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(1, out_features)) # -b
        self.V_auxiliary = nn.Parameter(torch.randn(in_features+1, out_features))
        self.reset_parameters(reset_params)
        self.activation = activation

    
    def reset_parameters(self, reset_params: str = "eye") -> None:
        in_features, out_features = self.U.shape
        if reset_params == "eye":
            if in_features <= out_features:
                with torch.no_grad():
                    self.U.data.copy_(0.5 * torch.eye(in_features, out_features))
            else:
                print("'eye' initialization not possible, defaulting to 'kaiming'")
                with torch.no_grad():
                    self.U.data.copy_(torch.randn(in_features, out_features) * (2 * in_features * out_features) ** -0.5)
            self.a.data.fill_(0.0)
        elif reset_params == "kaiming":
            with torch.no_grad():
                self.U.data.copy_(torch.randn(in_features, out_features) * (2 * in_features * out_features) ** -0.5)
            self.a.data.fill_(0.0)
        else:
            raise KeyError(f"Unkown reset_params value: {reset_params}")

    def create_spacelike_vector(self) -> torch.Tensor:
        U_norm = self.U.norm(dim=0, keepdim=True)
        U_norm_sqrt_k_b = self.manifold.k().sqrt() * U_norm * self.a
        time = - U_norm * torch.sinh(U_norm_sqrt_k_b)
        space = torch.cosh(U_norm_sqrt_k_b) * self.U
        return torch.cat([time, space], dim=0)
    
    def signed_dist2hyperplanes_scaled_angle(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the distances by scaling the angle (implicitely)"""
        V = self.create_spacelike_vector()
        sqrt_k = self.manifold.k().sqrt()
        return 1 / sqrt_k * torch.asinh(sqrt_k * x @ V)
    
    def signed_dist2hyperplanes_scaled_dist(self, x: torch.Tensor) -> torch.Tensor:
        """ Scale the distances by scaling the total distance (explicitly)"""
        V = self.create_spacelike_vector()
        V_norm = self.manifold.normL(V.transpose(0, 1)).transpose(0, 1)
        sqrt_k = self.manifold.k().sqrt()
        return V_norm / sqrt_k * torch.asinh(sqrt_k * x @ (V/V_norm))
    
    def compute_output_space(self, x: torch.Tensor) -> torch.Tensor:
        V = self.create_spacelike_vector()
        return self.activation(x @ V)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_space = self.compute_output_space(x)
        return self.manifold.projection_space_orthogonal(output_space)

    def forward_cache(self, x):
        output_space = self.activation(x @ self.V_auxiliary)
        return self.manifold.projection_space_orthogonal(output_space)

    def mlr(self, x):
        return self.signed_dist2hyperplanes_scaled_angle(x)
