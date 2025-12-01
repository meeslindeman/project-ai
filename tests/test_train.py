import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import logging

RED = "\033[91m"
YELLOW = "\033[93m"
PURPLE = "\033[95m"
RESET = "\033[0m"

def color_red(s):
    return f"{RED}{s}{RESET}"

def color_yellow(s):
    return f"{YELLOW}{s}{RESET}"

def color_purple(s):
    return f"{PURPLE}{s}{RESET}"

def make_batch(batch_size, seq_len, vocab_size, pad_id=0, special_token=1, device="cpu"):
    """
    Generate a batch of random sequences with padding.

    - token_ids: [B, N] with values in {0..vocab_size-1}, 0 = padding.
    - mask:      [B, N] bool, True for non-pad tokens.
    - labels:    [B], 0 or 1:
        label = 1 if special_token appears at least once in the sequence (excluding pads), else 0.
    """
    lengths = torch.randint(1, seq_len + 1, (batch_size,), device=device)

    token_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device=device)

    for i in range(batch_size):
        L = lengths[i].item()
        # sample tokens from [1..vocab_size-1] for non-pad positions
        token_ids[i, :L] = torch.randint(1, vocab_size, (L,), device=device)

    mask = token_ids != pad_id 

    # 1 if special_token appears at least once in the true tokens
    contains_special = (token_ids == special_token) & mask
    labels = contains_special.any(dim=1).long() 

    return token_ids, mask, labels

def check_param_nans(model, logger=None):
    """
    Inspect parameters and gradients for NaNs/Infs and log basic stats (mean, abs max).
    """
    log = logger.info if logger is not None else print

    for name, p in model.named_parameters():
        data = p.data

        data_nan = torch.isnan(data).any()
        data_inf = torch.isinf(data).any()
        data_mean = data.mean().item()
        data_absmax = data.abs().max().item()

        if data_nan or data_inf:
            log(color_red(
                f"[NaN/Inf param] {name} | nan={bool(data_nan)} | inf={bool(data_inf)}"
            ))

        log(
            f"[Param] {name:30s} | "
            f"w_mean={data_mean:+.3e} | w_absmax={data_absmax:.3e}"
        )

        if p.grad is not None:
            g = p.grad
            grad_nan = torch.isnan(g).any()
            grad_inf = torch.isinf(g).any()
            grad_mean = g.mean().item()
            grad_absmax = g.abs().max().item()

            if grad_nan or grad_inf:
                log(color_red(
                    f"[NaN/Inf grad ] {name} | nan={bool(grad_nan)} | inf={bool(grad_inf)}"
                ))

            log(
                f"[Grad ] {name:30s} | "
                f"g_mean={grad_mean:+.3e} | g_absmax={grad_absmax:.3e}"
            )
        else:
            log(f"[Grad ] {name:30s} | None")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Toy script to debug NaNs in classifier training."
    )

    # Core configuration
    parser.add_argument("--model", type=str, default="personal", choices=["personal", "hypformer"], help="Which model variant to use.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use, e.g. 'cpu' or 'cuda'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Data / model hyperparams
    parser.add_argument("--vocab-size", type=int, default=50, help="Vocabulary size.")
    parser.add_argument("--pad-id", type=int, default=0, help="Padding token id.")
    parser.add_argument("--embed-dim", type=int, default=8, help="Embedding dimension.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes.")
    parser.add_argument("--num-heads", type=int, default=1, help="Number of attention heads.")
    parser.add_argument("--curvature-k", type=float, default=0.1,
                        help="Curvature parameter k (for personal model).")
    parser.add_argument("--seq-len", type=int, default=6, help="Sequence length.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-steps", type=int, default=5, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")

    # Logging / debugging
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    parser.add_argument("--log-interval", type=int, default=1, help="How often to log training stats.")
    parser.add_argument("--trace-nans", action="store_true", help="If set, run detailed NaN/Inf + stats checks each step.")
    parser.add_argument("--attn-debug", action="store_true", help="Enable detailed debug logging inside LorentzAttention.")

    return parser.parse_args()

def main(args):
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=log_level,
    )
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed)

    device = args.device

    if args.model == "personal":
        from models.personal.model import Classifier
        model = Classifier(
            vocab_size=args.vocab_size,
            pad_id=args.pad_id,
            embed_dim=args.embed_dim,
            num_classes=args.num_classes,
            curvature_k=args.curvature_k,
            num_heads=args.num_heads,
            compute_scores="lorentz_inner",     # or "signed_dist"
            concat_operation="log-radius",      # or "log-radius"
            attn_debug=args.attn_debug
        ).to(device)
    elif args.model == "hypformer":
        from models.hypformer.model import Classifier
        model = Classifier(
            vocab_size=args.vocab_size,
            pad_id=args.pad_id,
            embed_dim=args.embed_dim,
            num_classes=args.num_classes,
            att_type="full",
            decoder_type="linear",
            num_heads=args.num_heads
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = model.double()

    model.train()

    for step in range(1, args.num_steps + 1):
        token_ids, mask, labels = make_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            pad_id=args.pad_id,
            special_token=1,
            device=device,
        )

        logits = model(token_ids, mask)

        # Forward pass sanity checks
        logits_nan = torch.isnan(logits).any()
        logits_inf = torch.isinf(logits).any()
        if logits_nan or logits_inf:
            logger.error(
                color_red(
                    f"NaN/Inf detected in LOGITS at step {step} | "
                    f"nan={bool(logits_nan)} | inf={bool(logits_inf)}"
                )
            )
            if args.trace_nans:
                check_param_nans(model, logger)
            # Early break to inspect state
            break

        loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                color_red(
                    f"NaN/Inf detected in LOSS at step {step} | loss={loss.item()}"
                )
            )
            if args.trace_nans:
                check_param_nans(model, logger)
            # Early break to inspect state
            break

        optimizer.zero_grad()
        loss.backward()

        if args.trace_nans:
            # Check params/gradients every step if tracing is enabled
            check_param_nans(model, logger)

        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        if step % args.log_interval == 0 or step == 1:
            logger.info(color_purple(
                f"Step {step:03d} | Loss: {loss.item():.6f} | "
                f"Acc: {acc * 100:.1f}% | loss_is_nan={torch.isnan(loss).item()}"
            ))

    model.eval()
    with torch.no_grad():
        token_ids, mask, labels = make_batch(
            batch_size=64,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            pad_id=args.pad_id,
            special_token=1,
            device=device,
        )
        logits = model(token_ids, mask)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()

if __name__ == "__main__":
    args = parse_args()
    main(args)

# NOTE: batch size = 1 seems to prevent nans
# NOTE: model.double() seems to prevent nans