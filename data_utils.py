import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse as sp  

"""
Data processing adapted from SGFormer.
"""

def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True):
    """Randomly split node indices into train/valid/test."""
    if label.dim() == 2:
        label_1d = label[:, 0]
    else:
        label_1d = label

    if ignore_negative:
        labeled_nodes = torch.where(label_1d != -1)[0]
    else:
        labeled_nodes = torch.arange(label_1d.shape[0])

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n), dtype=torch.long)

    train_idx = labeled_nodes[perm[:train_num]]
    valid_idx = labeled_nodes[perm[train_num: train_num + valid_num]]
    test_idx = labeled_nodes[perm[train_num + valid_num:]]
    return train_idx, valid_idx, test_idx

def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    """
    Sample label_num_per_class nodes per class for train; then sample valid/test from remaining.
    Expects labels to be [N] or [N,1].
    """
    if label.dim() == 2:
        label_1d = label[:, 0]
    else:
        label_1d = label

    train_idx, non_train_idx = [], []
    idx = torch.arange(label_1d.shape[0])
    class_list = label_1d.unique()

    for c in class_list.tolist():
        idx_c = idx[label_1d == c]
        perm = idx_c[torch.randperm(idx_c.numel())]
        train_idx += perm[:label_num_per_class].tolist()
        non_train_idx += perm[label_num_per_class:].tolist()

    train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    non_train_idx = torch.as_tensor(non_train_idx, dtype=torch.long)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.numel())]

    valid_idx = non_train_idx[:valid_num]
    test_idx = non_train_idx[valid_num: valid_num + test_num]
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def normalize_feat(mx):
    """Row-normalize np array or scipy sparse matrix."""
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.zeros_like(rowsum)
    nonzero = rowsum != 0
    r_inv[nonzero] = 1.0 / rowsum[nonzero]
    return sp.diags(r_inv).dot(mx)


def eval_acc(y_true, y_pred):
    """
    Node classification accuracy.
    """
    if y_true.dim() == 2:
        y_true = y_true[:, 0]
    y_true = y_true.detach().cpu().numpy()

    y_hat = y_pred.argmax(dim=-1).detach().cpu().numpy()
    return float((y_hat == y_true).mean())


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    """
    Returns: train_acc, valid_acc, test_acc, valid_loss, log_probs
    """
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(dataset)  # expected [N, C]

    train_acc = eval_func(dataset.label[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(dataset.label[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(dataset.label[split_idx["test"]], out[split_idx["test"]])

    logp = F.log_softmax(out, dim=1)
    y_valid = dataset.label[split_idx["valid"]]
    if y_valid.dim() == 2:
        y_valid = y_valid[:, 0]
    valid_loss = criterion(logp[split_idx["valid"]], y_valid)

    return train_acc, valid_acc, test_acc, valid_loss, logp


def load_fixed_splits(dataset, name, protocol=None):
    """
    Load fixed splits used by SGFormer: chameleon, squirrel, film.
    """
    splits_lst = []

    if name in ["chameleon", "squirrel"]:
        file_path = f"data/wiki_new/{name}/{name}_filtered.npz"
        data = np.load(file_path)
        train_masks = data["train_masks"]  # (10, N)
        val_masks = data["val_masks"]
        test_masks = data["test_masks"]
        N = train_masks.shape[1]
        node_idx = np.arange(N)

        for i in range(10):
            splits = {
                "train": torch.as_tensor(node_idx[train_masks[i]], dtype=torch.long),
                "valid": torch.as_tensor(node_idx[val_masks[i]], dtype=torch.long),
                "test":  torch.as_tensor(node_idx[test_masks[i]], dtype=torch.long),
            }
            splits_lst.append(splits)

    elif name in ["film"]:
        for i in range(10):
            splits_file_path = f"data/geom-gcn/{name}/{name}_split_0.6_0.2_{i}.npz"
            with np.load(splits_file_path) as sf:
                train_mask = torch.BoolTensor(sf["train_mask"])
                val_mask = torch.BoolTensor(sf["val_mask"])
                test_mask = torch.BoolTensor(sf["test_mask"])

            idx = torch.arange(train_mask.numel())
            splits = {
                "train": idx[train_mask],
                "valid": idx[val_mask],
                "test":  idx[test_mask],
            }
            splits_lst.append(splits)

    else:
        raise NotImplementedError(f"Fixed splits not implemented for dataset: {name}")

    return splits_lst