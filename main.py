import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from wordnet import build_dataset
from data import get_dataloader
from model import Classifier
from metrics import loss_fn, accuracy_fn

from config import Config

def train_epoch(model, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> tuple:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for token_ids, mask, labels in dataloader:
        token_ids, mask, labels = token_ids.to(device), mask.to(device), labels.to(device)

        logits = model(token_ids, mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = accuracy_fn(logits, labels)
        total_loss += loss.item()
        total_acc += acc
        steps += 1

    return total_loss / steps, total_acc / steps

@torch.no_grad()
def eval_epoch(model, dataloader: DataLoader, device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    for token_ids, mask, labels in dataloader:
        token_ids, mask, labels = token_ids.to(device), mask.to(device), labels.to(device)

        logits = model(token_ids, mask)
        loss = loss_fn(logits, labels)

        acc = accuracy_fn(logits, labels)
        total_loss += loss.item()
        total_acc += acc
        steps += 1

    return total_loss / steps, total_acc / steps

def main():
    cfg = Config()
    data = build_dataset()

    # fill model-dependent values from data
    cfg.vocab_size = data["vocab_size"]
    cfg.pad_id = data["pad_id"]
    cfg.num_classes = data["num_classes"]
    cfg.max_tokens = data["max_tokens"]

    # device setup
    if torch.cuda.is_available():
        cfg.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        cfg.device = torch.device("mps")
    else:
        cfg.device = torch.device("cpu")

    if cfg.device.type == "mps":  # warm up MPS if using Metal backend
        _ = torch.ones(1, device=cfg.device) * 1

    print(f"Using device: {cfg.device}")

    # dataloader
    train_loader = get_dataloader(
        data,
        batch_size = cfg.batch_size,
        shuffle = cfg.shuffle,
    )

    # model
    model = Classifier(
        vocab_size = cfg.vocab_size,
        pad_id = cfg.pad_id,
        embed_dim = cfg.dim_embed,
        num_classes = cfg.num_classes,
    ).to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # training loop
    for epoch in range(cfg.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, cfg.device)
        print({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
        })

if __name__ == "__main__":
    main()
