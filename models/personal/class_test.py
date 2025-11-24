import torch
import torch.nn as nn

from models.personal.model import Classifier
from manifolds.personal.lorentz import Lorentz


# -------------------------
#  Manifold checking utils
# -------------------------

def minkowski_norm_sq(x):
    t = x[..., :1]
    s = x[..., 1:]
    return -t * t + torch.sum(s * s, dim=-1, keepdim=True)

def is_on_manifold(x, manifold: Lorentz, tol=1e-4, verbose=True):
    k = manifold.k().item()           # curvature param
    target = -1.0 / k                 # expected <x,x>_L

    lnorm = minkowski_norm_sq(x)      # [..., 1]
    diff = (lnorm - target).abs().max().item()

    if verbose:
        print(f"[manifold check] k={k:.6f}")
        print(f"  target Minkowski norm = {target:.6f}")
        print(f"  max deviation = {diff:.6e}")

    return diff < tol


# -------------------------
#  Test Script
# -------------------------

def main():
    torch.manual_seed(42)

    # Small synthetic batch
    B = 2
    N = 5
    vocab_size = 30
    embed_dim = 8
    num_classes = 3
    num_heads = 2
    k = 0.1

    print("\nCreating PERSONAL Lorentz Classifier")
    model = Classifier(
        vocab_size=vocab_size,
        pad_id=0,
        embed_dim=embed_dim,
        num_classes=num_classes,
        k=k,
        num_heads=num_heads,
        compute_scores="lorentz_inner",
        value_agg="riemannian",
        concat_operation="direct",
    )

    # Generate random token batch
    token_ids = torch.randint(0, vocab_size, (B, N))
    mask = torch.ones(B, N, dtype=torch.bool)

    print("\nToken IDs:")
    print(token_ids)

    # ---- Step 1: Embeddings ----
    embeds = model.embedding(token_ids)
    print("\nEmbeddings:")
    print("  shape:", embeds.shape)

    # ---- Step 2: expmap0 ----
    manifold = model.manifold
    x_lorentz = manifold.expmap0(embeds)
    print("\nAfter expmap0 → Lorentz:")
    print("  shape:", x_lorentz.shape)

    print("\nChecking manifold validity of expmap0 output:")
    is_on_manifold(
        x_lorentz.reshape(-1, 1 + embed_dim),
        manifold=manifold,
        verbose=True
    )

    # ---- Step 3: Attention ----
    attn_out = model.attention(x_lorentz, mask)
    print("\nAttention Output:")
    print("  shape:", attn_out.shape)

    print("\nChecking manifold validity of attention output:")
    is_on_manifold(
        attn_out.reshape(-1, 1 + embed_dim),
        manifold=manifold,
        verbose=True
    )

    # ---- Step 4: Pooling ----
    pooled = model.pooling(attn_out, mask)
    print("\nPooled Output (Lorentz):")
    print("  shape:", pooled.shape)

    print("\nChecking manifold validity of pooled vectors:")
    is_on_manifold(
        pooled.reshape(-1, 1 + embed_dim),
        manifold=manifold,
        verbose=True
    )

    # ---- Step 5: Final classifier ----
    logits = model.fc(pooled)
    print("\nFinal logits (Euclidean):")
    print("  shape:", logits.shape)
    print("  logits:", logits)

    # Sample prints
    print("\nSample Lorentz point before classifier:")
    print(pooled[0])

    print("\nSample logits:")
    print(logits[0])


if __name__ == "__main__":
    main()
