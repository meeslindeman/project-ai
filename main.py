import argparse
import random
import numpy as np
import torch
import wandb
import torch.nn.functional as F

from dataset import load_nc_dataset
from data_utils import eval_acc, get_dataset_split
from geoopt.optim.radam import RiemannianAdam
from geoopt.optim.rsgd import RiemannianSGD


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
    print(f"Device: {device} | Using model: {args.model}")

    # Load dataset (works for both heterophilous and homophilous)
    dataset = load_nc_dataset(args)

    # Get appropriate splits based on dataset type
    split_idx = get_dataset_split(dataset, args)

    # Move tensors to device
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

    if args.model == "personal":
        if args.head_fusion != "midpoint":
            args.split_heads = True
            
        from models.personal.model import PersonalModel
        model = PersonalModel(
            input_dim=in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            compute_scores=args.compute_scores,
            head_fusion=args.head_fusion,
            split_heads=args.split_heads,
            curvature=args.curvature,
            reset_params=args.reset_params,
            a_default=args.a_default,
            attn_mask=attn_mask,
            dropout=args.dropout,
            use_ffn=args.use_ffn,
            train_curvature=args.train_curvature
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
            dropout=args.dropout,
            use_ffn=args.use_ffn,
            train_curvature=args.train_curvature
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
            dropout=args.dropout,
            lorentz_map=args.lorentz_map,
            use_ffn=args.use_ffn
        ).to(device)

    if args.precision:
        model = model.double()
        dataset.graph["node_feat"] = dataset.graph["node_feat"].double()

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "RiemannianAdam":
        optimizer = RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "RiemannianSGD":
        optimizer = RiemannianSGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    bad = 0

    print(f"Starting training for {args.dataset} (split {args.split})")

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

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch
            if args.train_curvature:
                best_curvature = model.manifold.k().item()
            else:
                best_curvature = None
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

        if args.wandb:
            wandb.log({
                "train/loss": loss.item(),
                "train/acc": train_acc,
                "val/acc": val_acc,
                "val/best_val": best_val,
                "test/acc": test_acc,
                "epoch": epoch
            })

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | loss {loss.item():.4f} | "
                f"train {train_acc:.4f} val {val_acc:.4f} test {test_acc:.4f}"
            )

        if bad >= args.patience:
            break

    print(f"Best val {best_val:.4f} at epoch {best_epoch}; corresponding test {best_test:.4f}")
    if args.train_curvature:
        print(f"Best curvature {best_curvature:.4f}")
    return best_test


def main(args):
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=f"{args.dataset}_{args.model}_h={args.num_heads}_lr={args.lr}",
            config=vars(args)
        )
        for k, v in wandb.config.items():
            setattr(args, k, v)

    train_one_split(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Node classification with hyperbolic models")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default="chameleon", 
                       choices=["chameleon", "squirrel", "actor", "airport", "disease"])
    
    # Dataset-specific args
    parser.add_argument("--no_feat_norm", action="store_true", help="Don't normalize features (chameleon/squirrel)")
    parser.add_argument("--normalize_feats", action="store_true", default=True, help="Normalize features (airport/disease)")
    parser.add_argument("--use_feats", action="store_true", default=True, help="Use node features (disease)")

    # Run control
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--split", type=int, default=0, help="Split index 0-9. Heterophilous: uses pre-defined splits. Homophilous: generates deterministic random split")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)

    # Model selection 
    parser.add_argument("--model", type=str, default="personal", choices=["personal", "hypformer", "euclidean"])
    parser.add_argument("--lorentz_map", action="store_true", help="Use Lorentz mapping for final classification layer (euclidean)")
    parser.add_argument("--use_ffn", action="store_true", help="Use FFN after each MHA layer")

    # Shared hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate on input features")
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--precision", action="store_true")
    parser.add_argument("--train_curvature", action="store_true")

    # Personal attention options
    parser.add_argument("--attn_scope", type=str, default="global", choices=["global", "adjs"])
    parser.add_argument("--compute_scores", type=str, default="lorentz_inner", choices=["lorentz_inner", "signed_dist"])
    parser.add_argument("--head_fusion", type=str, default="midpoint", choices=["midpoint", "concat_direct", "concat_logradius"])
    parser.add_argument("--split_heads", action="store_true", help="Split hidden dim across heads (D//H per head)")
    parser.add_argument("--reset_params", type=str, default="lorentz_kaiming", choices=["lorentz_kaiming", "kaiming"])
    parser.add_argument("--a_default", type=float, default=0.0)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="RiemannianAdam", choices=["Adam", "AdamW", "RiemannianAdam", "RiemannianSGD"])
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--wd", type=float, default=0.001)

    # Wandb args
    parser.add_argument('--wandb', nargs="?", const=True, default=False, help='Enable Weights & Biases logging')
    parser.add_argument("--wandb_project", type=str, default="hyperbolic-gnn")
    parser.add_argument('--wandb_entity', type=str, default="your-entity")
    parser.add_argument("--wandb_group", type=str, default="node-classification")

    args = parser.parse_args()

    if isinstance(args.wandb, str):
        args.wandb = args.wandb.lower() == "true"
        
    set_seed(args.seed)
    main(args)