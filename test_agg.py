import torch
import torch.nn as nn

from manifolds.personal import Lorentz, LorentzFC  # adjust import path
# from your_module import minkowski_norm_sq, is_on_manifold  # if you want checks


def hyperbolic_scores_times_v_with_lorentzfc_demo():
    torch.manual_seed(0)

    B, H, N = 1, 1, 3          # batch, heads, sequence length
    head_spatial_dim = 2       # spatial dim per head
    k = 0.1

    manifold = Lorentz(k)

    # ------------------------------------------------------------------
    # 1) Create fake Lorentz values v: [B, H, N, 1 + head_spatial_dim]
    # ------------------------------------------------------------------
    v_space = torch.randn(B, H, N, head_spatial_dim)          # spatial part
    v = manifold.projection_space_orthogonal(v_space)         # make them Lorentz points
    v_1 = manifold.expmap0(v_space)                           # alternative way
    
    print("v (Lorentz values):", v.shape)

    # ------------------------------------------------------------------
    # 2) Create attention weights attn: [B, H, N, N]
    # ------------------------------------------------------------------
    scores = torch.randn(B, H, N, N)
    attn = torch.softmax(scores, dim=-1)

    print("attn shape:", attn.shape)

    # ------------------------------------------------------------------
    # 3) Option 1: "true" hyperbolic aggregation:
    #    out_space = attn @ v_space, then project back to hyperboloid
    # ------------------------------------------------------------------
    v_space = v[..., 1:]                                         # [B,H,N,d]
    out_space_true = torch.einsum("bhnm,bhmd->bhnd", attn, v_space)
    out_true = manifold.projection_space_orthogonal(out_space_true)  # [B,H,N,1+d]

    print("out_true shape:", out_true.shape)

    # ------------------------------------------------------------------
    # 4) Option 2: abuse LorentzFC just to see what happens
    #
    # We take each attention row attn[0,0,i,:] in R^N,
    # map it to Lorentz space via expmap0, and feed it to LorentzFC.
    # NOTE: LorentzFC knows nothing about v, so its output ignores v entirely.
    # ------------------------------------------------------------------
    # LorentzFC expecting input Lorentz dim = N+1, so in_features = N
    fc = LorentzFC(
        in_features=N,
        out_features=head_spatial_dim,
        manifold=manifold,
        activation=nn.Identity(),
    )

    # Treat each attn row as a tangent vector in R^N
    # x_tan_attn: [N, N]  (one row per query position)
    x_tan_attn = attn[0, 0]                    # [N,N]
    x_lorentz_attn = manifold.expmap0(x_tan_attn)  # [N, N+1]

    # Pass through LorentzFC
    out_lfc = fc(x_lorentz_attn)              # [N, 1+head_spatial_dim]
    print("out_lfc shape:", out_lfc.shape)

    # ------------------------------------------------------------------
    # 5) Compare out_true[0,0] vs out_lfc
    # ------------------------------------------------------------------
    print("\n--- Comparison for each query position i ---")
    for i in range(N):
        print(f"\nQuery position i = {i}")
        print("  attn[i]:", attn[0, 0, i])
        print("  out_true[0,0,i]:", out_true[0, 0, i])
        print("  out_lfc[i]:      ", out_lfc[i])

    # Optional: max difference as a scalar
    diff = (out_true[0, 0] - out_lfc).abs().max().item()
    print("\nMax absolute difference between out_true[0,0] and out_lfc:", diff)


if __name__ == "__main__":
    hyperbolic_scores_times_v_with_lorentzfc_demo()
