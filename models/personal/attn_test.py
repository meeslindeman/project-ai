import torch
import torch.nn as nn

from models.personal.attention import LorentzAttention  # adjust path if needed
from manifolds.personal.lorentz import Lorentz


def minkowski_norm_sq(x: torch.Tensor) -> torch.Tensor:
    time = x[..., :1]   # First component (time)
    space = x[..., 1:]  # Remaining components (space)
    norm_sq = -time * time + torch.sum(space * space, dim=-1, keepdim=True)
    return norm_sq


def is_on_manifold(x: torch.Tensor, manifold: Lorentz, tol: float = 1e-4, verbose: bool = True) -> bool:
    k_val = manifold.k().item()          # e.g. ≈ 0.1
    target = -1.0 / k_val                # hyperboloid: <x,x>_L = -1/k

    norm_sq = minkowski_norm_sq(x)       # [..., 1]
    diff = norm_sq - target
    max_diff = diff.abs().max().item()

    if verbose:
        print(f"[manifold check] k={k_val:.6f}")
        print(f"  target Minkowski norm = {target:.6f}")
        print(f"  max |<x,x>_L - target| = {max_diff:.6e}")

    return max_diff < tol


def run_single_config(
    compute_scores: str,
    value_agg: str,
    concat_operation: str,
    *,
    B: int = 2,
    N: int = 4,
    spatial_dim: int = 8,
    num_heads: int = 2,
    k: float = 0.1,
):
    print("\n" + "=" * 80)
    print(f"Config: compute_scores={compute_scores}, value_agg={value_agg}, concat_operation={concat_operation}")
    print("=" * 80)

    lorentz_dim = spatial_dim + 1

    attn = LorentzAttention(
        input_dim=lorentz_dim,
        curvature=k,
        num_heads=num_heads,
        heads_concat=True,
        compute_scores=compute_scores,   # "lorentz_inner" or "signed_dist"
        value_agg=value_agg,             # "ambient" or "riemannian"
        concat_operation=concat_operation,  # "direct" or "log-radius"
        out_dim=None,                    # default: same spatial dim as input
    )

    # random tangent input
    x = torch.randn(B, N, spatial_dim)
    print("Input x (tangent):")
    print("  shape:", x.shape)

    # project to Lorentz manifold
    x = attn.manifold.expmap0(x)
    print("x_lorentz:")
    print("  shape:", x.shape)

    print("\nChecking whether input x is on manifold:")
    _ = is_on_manifold(x.reshape(-1, lorentz_dim), manifold=attn.manifold, verbose=True)

    # full mask (all tokens valid)
    mask = torch.ones(B, N, dtype=torch.bool)

    # forward pass
    out = attn(x, mask=mask)
    print("\nOutput:")
    print("  shape:", out.shape)

    # flatten and check manifold membership
    B_flat, D = out.reshape(-1, out.shape[-1]).shape
    print("\nChecking whether output is on manifold:")
    _ = is_on_manifold(out.reshape(-1, D), manifold=attn.manifold, verbose=True)

    # show a couple of samples
    print("\nSample input[0, 0, :]:", x[0, 0])
    print("Sample output[0, 0, :]:", out[0, 0])


def main():
    torch.manual_seed(42)

    compute_scores_opts = ["lorentz_inner", "signed_dist"]
    value_agg_opts = ["ambient", "riemannian"]
    concat_opts = ["direct", "log-radius"]

    for compute_scores in compute_scores_opts:
        for value_agg in value_agg_opts:
            for concat_operation in concat_opts:
                run_single_config(
                    compute_scores=compute_scores,
                    value_agg=value_agg,
                    concat_operation=concat_operation,
                )


if __name__ == "__main__":
    main()
