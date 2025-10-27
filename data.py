import torch
from torch.utils.data import Dataset, DataLoader

class WordNetDataset(Dataset):
    def __init__(self, data: dict) -> None:
        self.train_nodes = data["train_nodes"]
        self.labels = data["labels"]
        self.sample_tokens = data["sample_tokens"]
        self.sample_masks = data["sample_masks"]

    def __len__(self):
        return len(self.train_nodes)

    def __getitem__(self, idx: int) -> tuple:
        u_id = self.train_nodes[idx]

        token_ids = torch.tensor(self.sample_tokens[u_id], dtype=torch.long)
        mask = torch.tensor(self.sample_masks[u_id], dtype=torch.bool)
        label = torch.tensor(self.labels[u_id], dtype=torch.long)
        return token_ids, mask, label
    
def get_dataloader(data, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
    dataset = WordNetDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )
    return loader

