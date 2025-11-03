import torch
from torch.utils.data import DataLoader, Dataset
from dataset import prepare_dataset

class WordNetDataset(Dataset):
    def __init__(self, examples: list):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        token_ids, mask, labels = self.examples[idx]

        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def get_dataloader(batch_size: int = 32, test_size: float = 0.10, val_size: float = 0.10, num_workers: int = 0) -> tuple:
    """Prepare DataLoaders for training, validation, and testing."""
    data = prepare_dataset()
    examples = data['dataset']

    n_total = len(examples)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val

    train_examples = examples[:n_train]
    val_examples = examples[n_train:n_train + n_val]
    test_examples = examples[n_train + n_val:]

    train_dataset = WordNetDataset(train_examples)
    val_dataset = WordNetDataset(val_examples)
    test_dataset = WordNetDataset(test_examples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    dataset_info = {
        'vocab_size': data['vocab_size'],
        'num_classes': data['num_classes'],
        'class_names': data['class_names']
    }

    return train_loader, val_loader, test_loader, dataset_info