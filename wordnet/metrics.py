import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)

def accuracy_fn(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total

def f1_fn(logits: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> float:
    preds = torch.argmax(logits, dim=1)
    
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    return f1_score(targets_np, preds_np, average=average, zero_division=0)