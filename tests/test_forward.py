import argparse
import logging

import torch
import torch.nn as nn  

from models.personal.attention import LorentzAttention
from models.personal.model import Classifier
from manifolds.personal.lorentz import Lorentz

logger = logging.getLogger(__name__)

def minkowski_norm_sq(x: torch.Tensor) -> torch.Tensor:
    t = x[..., :1]
    s = x[..., 1:]
    return -t * t + torch.sum(s * s, dim=-1, keepdim=True)

def is_on_manifold(x: torch.Tensor, manifold: Lorentz, tol: float = 1e-4, name: str = "") -> bool:
    k = manifold.k().item()
    target = -1.0 / k
    lnorm = minkowski_norm_sq(x)
    diff = (lnorm - target).abs().max().item()

    logger.info(
        "[manifold check] %s | k=%.6f | target=%.6f | max_deviation=%.6e",
        name,
        k,
        target,
        diff,
    )

    return diff < tol


def run_attn_test(
    compute_scores: str,
    concat_operation: str,
    value_agg: str,
    B: int,
    N: int,
    spatial_dim: int,
    num_heads: int,
    curvature_k: float,
    attn_debug: bool,
) -> None:
    logger.info(
        "Running ATTN test | compute_scores=%s | value_agg=%s | concat_operation=%s",
        compute_scores,
        value_agg,
        concat_operation,
    )

    lorentz_dim = spatial_dim + 1

    attn = LorentzAttention(
        input_dim=lorentz_dim,
        curvature=curvature_k,
        num_heads=num_heads,
        compute_scores=compute_scores,     # "lorentz_inner" or "signed_dist"
        value_agg=value_agg,               # typically "riemannian"
        concat_operation=concat_operation, # "direct" or "log-radius"
        out_dim=None,
        debug=attn_debug,
    )

    # random tangent input
    x_tan = torch.randn(B, N, spatial_dim)
    logger.info("Input (tangent) shape: %s", tuple(x_tan.shape))

    # project to Lorentz manifold
    x = attn.manifold.expmap0(x_tan)
    logger.info("Input (Lorentz) shape: %s", tuple(x.shape))

    is_on_manifold(
        x.reshape(-1, lorentz_dim), manifold=attn.manifold, name="attention_input"
    )

    # full mask (all tokens valid)
    mask = torch.ones(B, N, dtype=torch.bool)

    # forward pass
    out = attn(x, mask=mask)
    logger.info("Attention output shape: %s", tuple(out.shape))

    B_flat, D = out.reshape(-1, out.shape[-1]).shape
    logger.info("Flattened output shape: (%d, %d)", B_flat, D)

    is_on_manifold(
        out.reshape(-1, D), manifold=attn.manifold, name="attention_output"
    )


def run_classifier_test(
    B: int,
    N: int,
    vocab_size: int,
    embed_dim: int,
    num_classes: int,
    num_heads: int,
    curvature_k: float,
    compute_scores: str,
    concat_operation: str,
    attn_debug: bool,
) -> None:
    logger.info("Running CLASSIFIER test")

    model = Classifier(
        vocab_size=vocab_size,
        pad_id=0,
        embed_dim=embed_dim,
        num_classes=num_classes,
        curvature_k=curvature_k,
        num_heads=num_heads,
        compute_scores=compute_scores,
        concat_operation=concat_operation,
        attn_debug=attn_debug,  
    )

    token_ids = torch.randint(0, vocab_size, (B, N))
    mask = torch.ones(B, N, dtype=torch.bool)

    logger.info("Token IDs shape: %s", tuple(token_ids.shape))
    logger.debug("Token IDs sample:\n%s", token_ids)

    # embeddings
    embeds = model.embedding(token_ids)
    logger.info("Embeddings shape: %s", tuple(embeds.shape))

    # project to Lorentz manifold
    manifold = model.manifold
    x_lorentz = manifold.expmap0(embeds)
    logger.info("After expmap0 → Lorentz shape: %s", tuple(x_lorentz.shape))

    is_on_manifold(
        x_lorentz.reshape(-1, 1 + embed_dim),
        manifold=manifold,
        name="classifier_expmap0_output",
    )

    # attention layer
    attn_out = model.attention(x_lorentz, mask)
    logger.info("Attention output shape: %s", tuple(attn_out.shape))

    is_on_manifold(
        attn_out.reshape(-1, 1 + embed_dim),
        manifold=manifold,
        name="classifier_attention_output",
    )

    # pooling
    pooled = model.pooling(attn_out, mask)
    logger.info("Pooled output (Lorentz) shape: %s", tuple(pooled.shape))

    is_on_manifold(
        pooled.reshape(-1, 1 + embed_dim),
        manifold=manifold,
        name="classifier_pooled_output",
    )

    # final classifier
    logits = model.fc(pooled)
    logger.info("Final logits shape: %s", tuple(logits.shape))
    logger.debug("Logits sample:\n%s", logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single forward-pass tests for LorentzAttention and Classifier.")
    parser.add_argument("--target", type=str, default="both", choices=["attn", "class", "both"], help="Which component to test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # shared / attn hyperparameters
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size B.")
    parser.add_argument("--seq-len", type=int, default=4, help="Sequence length N.")
    parser.add_argument("--spatial-dim", type=int, default=8, help="Spatial dim for attention.")
    parser.add_argument("--num-heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--curvature-k", type=float, default=0.1, help="Curvature parameter k.")

    # classifier-specific
    parser.add_argument("--vocab-size", type=int, default=30, help="Vocabulary size.")
    parser.add_argument("--embed-dim", type=int, default=8, help="Embedding dimension.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes.")

    # attention scoring/concat
    parser.add_argument(
        "--compute-scores",
        type=str,
        default="lorentz_inner",
        choices=["lorentz_inner", "signed_dist"],
        help="How to compute attention scores.",
    )
    parser.add_argument(
        "--concat-operation",
        type=str,
        default="direct",
        choices=["direct", "log-radius"],
        help="How to concatenate heads for LorentzAttention.",
    )
    parser.add_argument(
        "--value-agg",
        type=str,
        default="riemannian",
        choices=["riemannian"],
        help="Value aggregation mode (currently only 'riemannian' is implemented).",
    )

    # logging / debug
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )
    parser.add_argument(
        "--attn-debug",
        action="store_true",
        help="Enable debug logging inside LorentzAttention.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    torch.manual_seed(args.seed)

    if args.target in ("attn", "both"):
        run_attn_test(
            compute_scores=args.compute_scores,
            concat_operation=args.concat_operation,
            value_agg=args.value_agg,
            B=args.batch_size,
            N=args.seq_len,
            spatial_dim=args.spatial_dim,
            num_heads=args.num_heads,
            curvature_k=args.curvature_k,
            attn_debug=args.attn_debug,
        )

    if args.target in ("class", "both"):
        run_classifier_test(
            B=args.batch_size,
            N=args.seq_len,
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            num_classes=args.num_classes,
            num_heads=args.num_heads,
            curvature_k=args.curvature_k,
            compute_scores=args.compute_scores,
            concat_operation=args.concat_operation,
            attn_debug=args.attn_debug,
        )


if __name__ == "__main__":
    main()
