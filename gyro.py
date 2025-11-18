import torch

from manifolds.personal import Hyperboloid

manifold = Hyperboloid()

def mobius_scalarmul(t, x, c):
    """Exp_0(t * Log_0(x))"""
    return manifold.expmap0(t * manifold.logmap0(x, c), c)

c = 1.0

u1 = torch.randn(1,2)
u2 = torch.rand(1,2)

x1 = manifold.expmap0(u1, c)
x2 = manifold.expmap0(u2, c)

pooled = mobius_scalarmul(0.5, manifold.mobius_add(x1, x2, c), c)

print(pooled)