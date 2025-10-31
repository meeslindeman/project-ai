import math, torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

from mlr_layers import Euclidean_mlr, Lorentz_mlr_forward, Lorentz_mlr_dist, Lorentz_mlr_angle

torch.manual_seed(0)
device = "cpu"

# ---- toy datasets ----
def make_blobs(n_per_class=128, sep=2.5, std=0.5):
    c0 = torch.randn(n_per_class, 2)*std + torch.tensor([-sep, 0.0])
    c1 = torch.randn(n_per_class, 2)*std + torch.tensor([ sep, 0.0])
    X = torch.cat([c0, c1], 0)
    y = torch.cat([torch.zeros(n_per_class, dtype=torch.long),
                   torch.ones(n_per_class,  dtype=torch.long)], 0)
    return X, y

def make_concentric_rings(n_per_class=256, r_inner=1.0, r_outer=3.0, noise=0.1):
    theta_in  = 2*math.pi*torch.rand(n_per_class)
    theta_out = 2*math.pi*torch.rand(n_per_class)
    inner = torch.stack([r_inner*torch.cos(theta_in),  r_inner*torch.sin(theta_in)], 1)
    outer = torch.stack([r_outer*torch.cos(theta_out), r_outer*torch.sin(theta_out)], 1)
    inner += noise*torch.randn_like(inner)
    outer += noise*torch.randn_like(outer)
    X = torch.cat([inner, outer], 0)
    y = torch.cat([torch.zeros(n_per_class, dtype=torch.long),
                   torch.ones(n_per_class,  dtype=torch.long)], 0)
    return X, y

def make_hierarchical_dataset(n_super=2, n_sub=3, n_per_sub=50, spread_super=5.0, spread_sub=0.8):
    torch.manual_seed(0)
    X_list, y_super, y_sub = [], [], []
    for i in range(n_super):
        super_center = torch.randn(2) * spread_super
        for j in range(n_sub):
            sub_center = super_center + torch.randn(2) * spread_sub
            pts = sub_center + 0.3 * torch.randn(n_per_sub, 2)
            X_list.append(pts)
            y_super.append(torch.full((n_per_sub,), i))
            y_sub.append(torch.full((n_per_sub,), i * n_sub + j))
    X = torch.cat(X_list, 0)
    y_super = torch.cat(y_super, 0)
    y_sub = torch.cat(y_sub, 0)
    return X, y_super, y_sub

# ---- train/eval harness ----
def run_quick_test(model, X, y, steps=200, lr=0.1):
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
    return float(loss.item()), acc

def create_model(model_type, dim, num_classes):
    if model_type == 'euclidean':
        return Euclidean_mlr(dim, num_classes)
    elif model_type == 'forward':
        return Lorentz_mlr_forward(dim, num_classes)
    elif model_type == 'angle':
        return Lorentz_mlr_angle(dim, num_classes)
    elif model_type == 'dist':
        return Lorentz_mlr_dist(dim, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def test_models_on_data(dim, num_classes, X, y, model_type):
    eu_model = create_model('euclidean', dim, num_classes)
    lo_model = create_model(model_type, dim, num_classes)
    loss_eu, acc_eu = run_quick_test(eu_model, X, y)
    loss_lo, acc_lo = run_quick_test(lo_model, X, y)
    return (loss_eu, acc_eu), (loss_lo, acc_lo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlr_type', type=str, default='forward', help='Type of MLR (forward, angle, dist)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    Xb, yb = make_blobs()
    Xr, yr = make_concentric_rings()
    Xh, y_super, y_sub = make_hierarchical_dataset()

    # sanity: one pass forward/backward on blobs
    eu = create_model('euclidean', 2, 2)
    lo = create_model(args.mlr_type, 2, 2)
    for m in [eu, lo]:
        out = m(Xb)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        for p in m.parameters():
            p.grad = None

    # quick fits: blobs and rings
    (loss_b_eu, acc_b_eu), (loss_b_lo, acc_b_lo) = test_models_on_data(2, 2, Xb, yb, args.mlr_type)
    (loss_r_eu, acc_r_eu), (loss_r_lo, acc_r_lo) = test_models_on_data(2, 2, Xr, yr, args.mlr_type)

    # quick fits: hierarchical, coarse vs fine
    n_super = int(y_super.max().item() + 1)
    n_sub = int(y_sub.max().item() + 1)

    (loss_hc_eu, acc_hc_eu), (loss_hc_lo, acc_hc_lo) = test_models_on_data(2, n_super, Xh, y_super, args.mlr_type)
    (loss_hf_eu, acc_hf_eu), (loss_hf_lo, acc_hf_lo) = test_models_on_data(2, n_sub, Xh, y_sub, args.mlr_type)

    # report
    print(f"MLR type: {args.mlr_type}")
    print("===================================")
    print("Blobs  (linear-separable):")
    print(f"  Euclidean  -> loss {loss_b_eu:.3f}, acc {acc_b_eu:.3f}")
    print(f"  Lorentzian -> loss {loss_b_lo:.3f}, acc {acc_b_lo:.3f}")

    print("Rings  (nonlinear):")
    print(f"  Euclidean  -> loss {loss_r_eu:.3f}, acc {acc_r_eu:.3f}")
    print(f"  Lorentzian -> loss {loss_r_lo:.3f}, acc {acc_r_lo:.3f}")

    print("Hierarchical coarse (super-clusters):")
    print(f"  Euclidean  -> loss {loss_hc_eu:.3f}, acc {acc_hc_eu:.3f}")
    print(f"  Lorentzian -> loss {loss_hc_lo:.3f}, acc {acc_hc_lo:.3f}")

    print("Hierarchical fine (sub-clusters):")
    print(f"  Euclidean  -> loss {loss_hf_eu:.3f}, acc {acc_hf_eu:.3f}")
    print(f"  Lorentzian -> loss {loss_hf_lo:.3f}, acc {acc_hf_lo:.3f}")