import torch
import torch.nn as nn
import torch.optim as optim

from manifolds.personal.layer import LorentzFC, LorentzMLR
from manifolds.personal.lorentz import Lorentz

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

class Classifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 4, depth: int = 4, num_classes: int = 2, k: float = 0.1, pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.manifold = Lorentz(k)
        self.layers = nn.ModuleList([
            LorentzFC(
                in_features=embed_dim,   # spatial dim D
                out_features=embed_dim,  # keep same D
                manifold=self.manifold,
                reset_params="kaiming"
            )
            for _ in range(depth)
        ])

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)             # [B, N, D]

        mask_f = mask.unsqueeze(-1).float()         # [B, N, 1]
        sum_emb = (emb * mask_f).sum(dim=1)         # [B, D]
        lengths = mask_f.sum(dim=1).clamp_min(1.0)  # [B, 1]
        avg_emb = sum_emb / lengths                 # [B, D] in tangent space

        x = self.manifold.expmap0(avg_emb)          # [B, D+1]

        for layer in self.layers:
            x = layer(x)                            # stays [B, D+1]

        tangent = self.manifold.logmap0(x)          # [B, D]

        logits = self.classifier(tangent)           # [B, 2]
        return logits

def train_one_epoch(model, optimizer, device, batch_size, seq_len, vocab_size, steps_per_epoch=100):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for _ in range(steps_per_epoch):
        token_ids, mask, labels = make_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device
        )

        logits = model(token_ids, mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, device, batch_size, seq_len, vocab_size, steps=50):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for _ in range(steps):
        token_ids, mask, labels = make_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device
        )
        logits = model(token_ids, mask)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    vocab_size = 50
    seq_len = 32
    batch_size = 128
    embed_dim = 4
    lr = 1e-3
    num_epochs = 10
    depths = [1, 2, 4, 8]

    print(f"Device: {device}")
    print(f"Toy task: detect presence of special token (1) in sequence.")
    print(f"Seq len = {seq_len}, vocab_size = {vocab_size}, embed_dim = {embed_dim}")
    print(f"Depths to test: {depths}")
    print("-" * 80)

    for depth in depths:
        print(f"\n{'=' * 60}")
        print(f"Training Classifier with depth = {depth}")
        print(f"{'=' * 60}")

        model = Classifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            depth=depth,
            num_classes=2,
            k=0.1,
            pad_id=0
        ).to(device)
        # model = model.double()

        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, optimizer, device,
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                steps_per_epoch=100
            )
            val_loss, val_acc = evaluate(
                model, device,
                batch_size=batch_size,
                seq_len=seq_len,
                vocab_size=vocab_size,
                steps=50
            )
            best_acc = max(best_acc, val_acc)

            print(
                f"[Depth {depth:2d}] Epoch {epoch:02d}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:6.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:6.2f}% "
                f"(Best: {best_acc*100:6.2f}%)"
            )


if __name__ == "__main__":
    main()
