import torch
import torch.nn as nn

from models.personal.attention import LorentzAttention  # adjust path if needed


def sample_lorentz_points(batch_size: int, seq_len: int, dim: int) -> torch.Tensor:
    """
    Sample points on the upper sheet of the Lorentz hyperboloid:
    - spatial part ~ N(0, 1)
    - time = sqrt(1 + ||x||^2)  (curvature -1)
    Returns: (B, N, dim), where dim = 1 (time) + (dim-1) (space)
    """
    assert dim >= 2, "Need at least 1 time + 1 space dimension"

    # spatial: (B, N, dim-1)
    x_space = torch.randn(batch_size, seq_len, dim - 1)

    # norm^2 of spatial part
    sq_norm = (x_space ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)

    # time coordinate on upper sheet
    t = torch.sqrt(1.0 + sq_norm)

    # concatenate time and space
    x = torch.cat([t, x_space], dim=-1)  # (B, N, dim)
    return x


def minkowski_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Minkowski "norm" (Lorentzian squared distance to origin):
    〈x, x〉_L = -t^2 + ||x_space||^2
    x: (..., D) with x[..., 0] = time, x[..., 1:] = space
    Returns: (...,) tensor of norms.
    """
    t = x[..., 0]
    x_space = x[..., 1:]
    return -t * t + (x_space ** 2).sum(dim=-1)


def is_on_hyperboloid(x: torch.Tensor, atol: float = 1e-4) -> torch.Tensor:
    """
    Check if points lie on the hyperboloid -t^2 + ||x||^2 = -1 (curvature -1).
    Returns a boolean mask with same leading shape as x[..., 0].
    """
    mn = minkowski_norm(x)
    return (mn + 1.0).abs() <= atol


def main():
    torch.manual_seed(0)

    B = 2        # batch size
    N = 4        # number of tokens
    D = 5        # Lorentz dimension: 1 time + 4 space
    H = 1        # number of heads

    # 1. Sample input points on the Lorentz manifold
    x = sample_lorentz_points(B, N, D)  # (B, N, D)

    print("Input Minkowski norms:", minkowski_norm(x))
    print("All input on manifold:", bool(is_on_hyperboloid(x).all()))

    # 2. Dummy mask (all tokens valid)
    mask = torch.ones(B, N, dtype=torch.bool)

    # 3. Instantiate attention
    attn = LorentzAttention(input_dim=D, num_heads=H, heads_concat=False)

    # 4. Forward pass
    out = attn(x, mask)  # expected shape: (B, N, D) or similar

    print("Output shape:", out.shape)
    print("Output Minkowski norms:", minkowski_norm(out))

    on_manifold_mask = is_on_hyperboloid(out)
    print("All output on manifold:", bool(on_manifold_mask.all()))
    print("Fraction on manifold:", on_manifold_mask.float().mean().item())


if __name__ == "__main__":
    main()
