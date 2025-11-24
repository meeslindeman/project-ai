import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
import argparse

from experiments.wordnet.dataloader import get_dataloader
from experiments.wordnet.metrics import loss_fn, accuracy_fn, f1_fn

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> tuple:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    steps = 0

    for batch in dataloader:
        token_ids = batch['token_ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(token_ids, mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = accuracy_fn(logits, labels)
            f1 = f1_fn(logits, labels)
        
        total_loss += loss.item()
        total_acc += acc
        total_f1 += f1
        steps += 1

    return total_loss / steps, total_acc / steps, total_f1 / steps

@torch.no_grad()
def eval_epoch(model, dataloader: DataLoader, device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    steps = 0

    for batch in dataloader:
        token_ids = batch['token_ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(token_ids, mask)
        loss = loss_fn(logits, labels)
        acc = accuracy_fn(logits, labels)
        f1 = f1_fn(logits, labels)
        
        total_loss += loss.item()
        total_acc += acc
        total_f1 += f1
        steps += 1

    return total_loss / steps, total_acc / steps, total_f1 / steps

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, info = get_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    vocab_size = info['vocab_size']
    num_classes = info['num_classes']

    print(f"\nLoading model: {args.model}")
    if args.model == 'euclidean':
        from models.euclidean import Classifier
        model = Classifier(
            vocab_size=vocab_size,
            pad_id=0,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            num_heads=args.num_heads
        ).to(device)
    elif args.model == 'hypformer':
        print(f"Model specifics: {args.att_type} | {args.decoder}")
        from models.hypformer.model import Classifier
        model = Classifier(
            vocab_size=vocab_size,
            pad_id=0,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            att_type=args.att_type,
            decoder_type=args.decoder,
            num_heads=args.num_heads
        ).to(device)
    elif args.model == 'personal':
        from models.personal.model import Classifier
        model = Classifier(
            vocab_size=vocab_size,
            pad_id=0,
            embed_dim=args.embed_dim,
            num_classes=num_classes,
            num_heads=args.num_heads,
            compute_scores=args.compute_scores,      
            value_agg=args.value_agg,
            concat_operation=args.concat_operation
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     torch.save(model.state_dict(), 'best_model.pt')
        
        # print("\nEvaluating on test set...")
        # model.load_state_dict(torch.load('best_model.pt'))
        # test_loss, test_acc = eval_epoch(model, test_loader, device)
        # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train WordNet Subtree Classifier')
    
    # Data args
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    
    # Model args
    parser.add_argument('--model', type=str, default="euclidean", help="Which model to train")
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=1, help="Number of attention heads")

    # Hypformer
    parser.add_argument('--att_type', type=str, default="full", help="('full', 'focus_attention')")
    parser.add_argument('--decoder', type=str, default="linear", help="('cls', 'linear')")

    # Personal model
    parser.add_argument("--compute_scores", type=str, default="lorentz_inner")
    parser.add_argument("--value_agg", type=str, default="riemannian")
    parser.add_argument("--concat_operation", type=str, default="direct")
    
    # Training args
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    main(args)