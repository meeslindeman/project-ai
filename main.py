import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from dataset import load_nc_dataset
from data_utils import load_fixed_splits, eval_acc
from hypll.optim import RiemannianAdam


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


@torch.no_grad()
def evaluate_split(model, dataset, split_idx, device):
    model.eval()
    x = dataset.graph["node_feat"]
    out = model(x)  # [N, C]
    y = dataset.label.to(device)

    train_acc = eval_acc(y[split_idx["train"]], out[split_idx["train"]])
    val_acc = eval_acc(y[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_acc(y[split_idx["test"]], out[split_idx["test"]])
    return train_acc, val_acc, test_acc


def train_one_split(args):
    device = get_device()
    print(f"Device: {device}")

    dataset = load_nc_dataset(args)

    # fixed splits for chameleon/squirrel/film
    name = args.dataset
    if name == "actor":
        name = "film"
    splits = load_fixed_splits(dataset, name=name, protocol=None)
    split_idx = splits[args.split]

    # move tensors to device
    dataset.graph["node_feat"] = dataset.graph["node_feat"].to(device)
    dataset.graph["edge_index"] = dataset.graph["edge_index"].to(device)
    dataset.label = dataset.label.to(device)

    in_dim = dataset.graph["node_feat"].shape[1]
    num_classes = int(dataset.label.max().item() + 1)

    attn_mask = None
    if args.attn_scope == "adjs":
        N = dataset.graph["num_nodes"]
        edge_index = dataset.graph["edge_index"]
        attn_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        attn_mask[edge_index[0], edge_index[1]] = True
        attn_mask.fill_diagonal_(True)

    if args.no_split_heads:
        args.split_heads = False
    elif args.split_heads:
        args.split_heads = True
    else:
        # default behavior
        args.split_heads = True

    print(vars(args))

    if args.model == "personal":
        if args.concat_operation in ("direct", "log-radius") and not args.split_heads:
            raise ValueError("For personal model: direct/log-radius require --split_heads (do not use --no_split_heads).")
        if args.concat_operation == "none" and args.split_heads:
            raise ValueError("For personal model: concat_operation=none requires --no_split_heads (HypFormer-like heads).")

        from models.personal.model import GraphClassifier
        model = GraphClassifier(
            input_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            curvature_k=args.curvature,
            num_heads=args.num_heads,
            compute_scores=args.compute_scores,
            concat_operation=args.concat_operation,
            split_heads=args.split_heads,
            attn_debug=args.attn_debug,
            num_layers=args.num_layers,
            attn_mask=attn_mask,
        ).to(device)

    elif args.model == "hypformer":
        from models.hypformer.model import Hypformer
        model = Hypformer(
            input_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            att_type="full",
            decoder_type="linear",
            num_heads=args.num_heads,
            k=args.curvature,
            attn_mask=attn_mask,
            alpha=args.alpha,
            dropout=args.dropout
        ).to(device)

    elif args.model == "euclidean":
        from models.euclidean.model import EuclideanModel
        model = EuclideanModel(
            input_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            attn_mask=attn_mask,
            alpha=args.alpha,
            lorentz_map=args.lorentz_map
        ).to(device)

    if args.precision:
        model = model.double()
        dataset.graph["node_feat"] = dataset.graph["node_feat"].double()

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "RiemannianAdam":
        if RiemannianAdam is None:
            raise RuntimeError("hypll.optim.RiemannianAdam not available in this environment.")
        optimizer = RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        x = dataset.graph["node_feat"]
        out = model(x)  # [N, C]

        y = dataset.label
        if y.dim() == 2:
            y = y[:, 0]  # [N]

        loss = F.cross_entropy(out[split_idx["train"]], y[split_idx["train"]])
        if not torch.isfinite(loss):
            print(f"Non-finite loss at epoch {epoch}, aborting.")
            break

        loss.backward()
        optimizer.step()

        train_acc, val_acc, test_acc = evaluate_split(model, dataset, split_idx, device)

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | loss {loss.item():.4f} | "
                f"train {train_acc:.4f} val {val_acc:.4f} test {test_acc:.4f}"
            )

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    print(f"Best val {best_val:.4f} at epoch {best_epoch}; corresponding test {best_test:.4f}")
    return best_test


def main():
    parser = argparse.ArgumentParser("Node classification (heterophily)")

    # dataset
    parser.add_argument("--dataset", type=str, default="chameleon", choices=["chameleon", "squirrel", "actor"])
    parser.add_argument("--no_feat_norm", action="store_true")

    # run control
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=int, default=0, help="0..9 for fixed splits")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)

    # model selection 
    parser.add_argument("--model", type=str, default="personal", choices=["personal", "hypformer", "euclidean", "euclidean_map"])
    parser.add_argument("--lorentz_map", action="store_true", help="Use Lorentz mapping for final classification layer (euclidean)")

   # shared hparams
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0, help="Residual connection weight")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (not fully implemented)")
    parser.add_argument("--curvature", type=float, default=0.1)
    parser.add_argument("--precision", action="store_true")

    # personal attention options
    parser.add_argument("--attn_scope", type=str, default="global", choices=["global", "adjs"])
    parser.add_argument("--compute_scores", type=str, default="lorentz_inner", choices=["lorentz_inner", "signed_dist"])
    parser.add_argument("--concat_operation", type=str, default="direct", choices=["direct", "log-radius", "none"])
    parser.add_argument("--split_heads", action="store_true", help="(personal) Split hidden dim across heads (D//H per head). Required for direct/log-radius.")
    parser.add_argument("--no_split_heads", action="store_true", help="(personal) Do NOT split dim across heads (D per head). Use with concat_operation=none for HypFormer-like.")
    parser.add_argument("--attn_debug", action="store_true")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="RiemannianAdam", choices=["Adam", "RiemannianAdam"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-3)

    args = parser.parse_args()
    set_seed(args.seed)
    train_one_split(args)


if __name__ == "__main__":
    main()