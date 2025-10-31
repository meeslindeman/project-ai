import torch
import torch.nn as nn
from lorentz import Lorentz
from lorentz_fc import LorentzFC

class Euclidean_mlr(nn.Module):
    """A fully connected layer in the Euclidean model with identity activation."""
    # the activation is the identity
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)

class Lorentz_mlr_forward(nn.Module):
    """A fully connected layer in the Lorentz model. with identity activation."""
    # the activation is the identity
    def __init__(self, in_features: int, out_features: int, k: float = 0.1, reset_params: str = "kaiming", activation: nn.Module = nn.Identity()) -> None:
        super().__init__()
        self.manifold = Lorentz(k)
        self.linear = LorentzFC(in_features, out_features, self.manifold, reset_params, activation)
    
    def forward(self, x):
        x = self.manifold.expmap0(x)
        return self.linear.compute_output_space(x) 

class Lorentz_mlr_angle(nn.Module):
    """ A fully connected layer in the Lorentz model. with identity activation.
    """
    # the activation is the identity
    def __init__(self, in_features: int, out_features: int, k: float = 0.1, reset_params: str = "kaiming", activation: nn.Module = nn.Identity()) -> None:
        super().__init__()
        self.manifold = Lorentz(k)
        self.linear = LorentzFC(in_features, out_features, self.manifold, reset_params, activation)
    
    def forward(self, x):
        x = self.manifold.expmap0(x)
        return self.linear.signed_dist2hyperplanes_scaled_angle(x)

class Lorentz_mlr_dist(nn.Module):
    """ A fully connected layer in the Lorentz model. with identity activation.
    """
    # the activation is the identity
    def __init__(self, in_features: int, out_features: int, k: float = 0.1, reset_params: str = "kaiming", activation: nn.Module = nn.Identity()) -> None:
        super().__init__()
        self.manifold = Lorentz(k)
        self.linear = LorentzFC(in_features, out_features, self.manifold, reset_params, activation)
    
    def forward(self, x):
        x = self.manifold.expmap0(x)
        return self.linear.signed_dist2hyperplanes_scaled_dist(x)


"""
Lorentz_fully_connected(10, 5, manifold=Lorentz(0.1)) # accepts input of shape [batch_size, 10+1] and outputs [batch_size, 5+1]

...
'lorentz_dist': Lorentz_fully_connected_mlr_dist

model_class = MODEL_REGISTRY['lorentz_dist']
model = model_class(
    in_features=2048,
    out_features=num_classes,
    k=args.curvature
).to(device)

...
features, labels = features.to(device), labels.to(device)
outputs = model(features)
"""