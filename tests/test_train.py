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

def make_batch(batch_size, seq_len, vocab_size, pad_id=0, num_classes=4, device="cpu"):
    assert vocab_size > 1, "vocab_size must be > 1 (0 reserved for pad)."
    assert num_classes >= 2, "num_classes must be >= 2."

    lengths = torch.randint(1, seq_len + 1, (batch_size,), device=device)

    token_ids = torch.full(
        (batch_size, seq_len),
        pad_id,
        dtype=torch.long,
        device=device,
    )

    non_pad_vocab = vocab_size - 1
    group_size = max(1, non_pad_vocab // num_classes)

    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    # choose majority cluster for each sequence
    majority_c = torch.randint(0, num_classes, (batch_size,), device=device)

    all_tokens = torch.arange(1, vocab_size, device=device)

    for b in range(batch_size):
        L = lengths[b].item()
        c = majority_c[b].item()

        # tokens belonging to cluster c
        start = 1 + c * group_size
        end = min(1 + (c + 1) * group_size, vocab_size)
        class_tokens = torch.arange(start, end, device=device)
        if class_tokens.numel() == 0:
            class_tokens = all_tokens

        # tokens from other clusters for noise
        other_tokens = all_tokens[(all_tokens < start) | (all_tokens >= end)]
        if other_tokens.numel() == 0:
            other_tokens = class_tokens

        p_main = 0.9
        use_main = torch.rand(L, device=device) < p_main

        main_choices = class_tokens[
            torch.randint(0, class_tokens.numel(), (L,), device=device)
        ]
        noise_choices = other_tokens[
            torch.randint(0, other_tokens.numel(), (L,), device=device)
        ]

        tokens = torch.where(use_main, main_choices, noise_choices)
        token_ids[b, :L] = tokens

        labels[b] = c

    mask = token_ids != pad_id
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
    parser.add_argument("--vocab-size", type=int, default=64, help="Vocabulary size (including pad).")
    parser.add_argument("--pad-id", type=int, default=0, help="Padding token id.")
    parser.add_argument("--embed-dim", type=int, default=8, help="Embedding dimension.")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of top-level classes (clusters).")
    parser.add_argument("--num-heads", type=int, default=1, help="Number of attention heads.")
    parser.add_argument("--curvature-k", type=float, default=0.1, help="Curvature parameter k (for personal model).")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer: Adam or SGD.")
    parser.add_argument("--precision", action="store_true", help="Enable float64 for model + loss.")
    parser.add_argument("--p-noise", type=float, default=0.1, help="Probability of sampling tokens from other classes (noise).")


    # Logging / debugging
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR.")
    parser.add_argument("--log-interval", type=int, default=20, help="How often to log training stats.")
    parser.add_argument("--trace-nans", action="store_true", help="If set, run detailed NaN/Inf + stats checks each step.")
    parser.add_argument("--attn-debug", action="store_true", help="Enable detailed debug logging inside LorentzAttention.")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="If > 0, apply gradient norm clipping with this max norm.")
    parser.add_argument("--val-size", type=int, default=512, help="Validation set size for stable metrics.")

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
            value_agg="midpoint",
            concat_operation="direct",      # or "log-radius"
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

    if args.precision:
        model = model.double()
        criterion = criterion.double()

    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown Optimizer")

    model.train()

    val_token_ids, val_mask, val_labels = make_batch(
        batch_size=args.val_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        pad_id=args.pad_id,
        num_classes=args.num_classes,
        device=device,
    )

    if args.precision:
        val_token_ids = val_token_ids.long()  # indices stay long
        val_mask = val_mask.bool()
        val_labels = val_labels.long()

    model.train()

    for step in range(1, args.num_steps + 1):
        token_ids, mask, labels = make_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            pad_id=args.pad_id,
            num_classes=args.num_classes,
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
            break

        optimizer.zero_grad()
        loss.backward()

        # Optional gradient clipping
        if args.clip_grad_norm and args.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)

        if args.trace_nans:
            check_param_nans(model, logger)

        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            train_acc = (preds == labels).float().mean().item()

        if step % args.log_interval == 0 or step == 1:
            logger.info(color_purple(
                f"Step {step:03d} | Train Loss: {loss.item():.6f} | "
                f"Train Acc: {train_acc * 100:.1f}% | loss_is_nan={torch.isnan(loss).item()}"
            ))

            # Validation metrics on fixed dataset
            model.eval()
            with torch.no_grad():
                val_logits = model(val_token_ids, val_mask)
                val_loss = criterion(val_logits, val_labels)
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (val_preds == val_labels).float().mean().item()
            model.train()

            logger.info(
                f"Val Loss: {val_loss.item():.6f} | Acc: {val_acc * 100:.1f}%"
            )

    # Final evaluation on a fresh batch (optional)
    model.eval()
    with torch.no_grad():
        token_ids, mask, labels = make_batch(
            batch_size=64,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            pad_id=args.pad_id,
            num_classes=args.num_classes,
            device=device,
        )
        logits = model(token_ids, mask)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        logging.info(f"Final fresh-batch acc: {acc * 100:.1f}%")

if __name__ == "__main__":
    args = parse_args()
    main(args)
