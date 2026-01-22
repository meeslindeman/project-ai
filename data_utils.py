import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


# ======================== Common Utilities ========================

def normalize_feat(mx):
    """Row-normalize np array or scipy sparse matrix."""
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.zeros_like(rowsum)
    nonzero = rowsum != 0
    r_inv[nonzero] = 1.0 / rowsum[nonzero]
    return sp.diags(r_inv).dot(mx)


def normalize(mx):
    """Row-normalize (potentially sparse) matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def bin_feat(feat, bins):
    """Bin continuous features into discrete bins."""
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def augment(adj, features, normalize_feats=True):
    """Augment features with degree information (for Airport dataset)."""
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ======================== Evaluation ========================

def eval_acc(y_true, y_pred):
    """Node classification accuracy."""
    if y_true.dim() == 2:
        y_true = y_true[:, 0]
    y_true = y_true.detach().cpu().numpy()

    y_hat = y_pred.argmax(dim=-1).detach().cpu().numpy()
    return float((y_hat == y_true).mean())


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    """
    Comprehensive evaluation function.
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


# ======================== Train/Val/Test Splits ========================

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


def split_data(labels, val_prop, test_prop, seed):
    """
    Split data for homophilous datasets (balanced pos/neg split).
    Used by Airport and Disease datasets.
    """
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[nb_val + nb_test:]
    
    train_idx = torch.tensor(idx_train_pos + idx_train_neg, dtype=torch.long)
    val_idx = torch.tensor(idx_val_pos + idx_val_neg, dtype=torch.long)
    test_idx = torch.tensor(idx_test_pos + idx_test_neg, dtype=torch.long)
    
    return {"train": train_idx, "valid": val_idx, "test": test_idx}


def load_fixed_splits(dataset, name, protocol=None):
    """
    Load fixed splits for heterophilous datasets: chameleon, squirrel, film.
    Returns list of 10 splits, each with keys: train, valid, test.
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


def get_dataset_split(dataset, args):
    """
    Unified split getter for both heterophilous and homophilous datasets.
    
    For heterophilous (chameleon, squirrel, actor): uses fixed splits (0-9)
    For homophilous (airport, disease): generates deterministic random splits
        - Split index (0-9) is combined with base seed to ensure reproducibility
        - Each split index gives a different random split, but reproducible
    
    Returns: split dict with keys: train, valid, test
    """
    name = args.dataset
    if name == "actor":
        name = "film"
    
    # Heterophilous datasets use fixed splits
    if name in ["chameleon", "squirrel", "film"]:
        splits = load_fixed_splits(dataset, name=name, protocol=None)
        if args.split >= 10:
            raise ValueError(f"Heterophilous datasets only have splits 0-9, got {args.split}")
        return splits[args.split]
    
    # Homophilous datasets use random splits with deterministic seeding
    elif name == "airport":
        val_prop, test_prop = 0.15, 0.15
        # Combine base seed with split index for deterministic but different splits
        split_seed = args.seed + args.split * 1000
        return split_data(dataset.label.numpy().flatten(), val_prop, test_prop, seed=split_seed)
    
    elif name == "disease":
        val_prop, test_prop = 0.10, 0.60
        # Combine base seed with split index for deterministic but different splits
        split_seed = args.seed + args.split * 1000
        return split_data(dataset.label.numpy().flatten(), val_prop, test_prop, seed=split_seed)
    
    else:
        raise ValueError(f"Unknown dataset for split generation: {args.dataset}")


# ======================== Legacy Functions (kept for compatibility) ========================

def mask_edges(adj, val_prop, test_prop, seed):
    """Split edges for link prediction (legacy, kept for compatibility)."""
    np.random.seed(seed)
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(test_edges_false)