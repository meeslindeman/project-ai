import torch
from torch.utils.data import Dataset, DataLoader
import random

class WordNetDataset(Dataset):
    def __init__(self, node_ids: list,data: dict) -> None:
        self.node_ids = node_ids
        self.labels = data["labels"]
        self.sample_tokens = data["sample_tokens"]
        self.sample_masks = data["sample_masks"]

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx: int) -> tuple:
        u_id = self.node_ids[idx]

        token_ids = torch.tensor(self.sample_tokens[u_id], dtype=torch.long)
        mask = torch.tensor(self.sample_masks[u_id], dtype=torch.bool)
        label = torch.tensor(self.labels[u_id], dtype=torch.long)
        return token_ids, mask, label

def get_dataloader(data, batch_size: int = 128, shuffle: bool = True, val_frac: float = 0.2) -> DataLoader:
    node_ids = data["nodes"]
    random.shuffle(node_ids)

    split_idx = int((1.0 - val_frac) * len(node_ids))
    train_ids = node_ids[:split_idx]
    val_ids = node_ids[split_idx:]

    train_ds = WordNetDataset(train_ids, data)
    val_ds = WordNetDataset(val_ids, data)

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader

