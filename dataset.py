import os
from os import path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from data_utils import normalize_feat, rand_train_test_idx

"""
Data processing adapted from SGFormer.
"""

DATAPATH = "data/"

class NCDataset(object):
    def __init__(self, name, root=DATAPATH):
        self.name = name
        self.root = root
        self.graph = {}
        self.label = None  # expected shape [N, 1] 

    def get_idx_split(self, split_type="random", train_prop=0.5, valid_prop=0.25):
        if split_type != "random":
            raise NotImplementedError(f"split_type={split_type} not supported here")

        ignore_negative = False if self.name == "ogbn-proteins" else True
        train_idx, valid_idx, test_idx = rand_train_test_idx(
            self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative
        )
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"


def load_nc_dataset(args):
    """
    args.dataset: 'chameleon' | 'squirrel' | 'film' (Actor)
    args.no_feat_norm: bool
    """
    global DATAPATH
    DATAPATH = getattr(args, "data_dir", "data/")

    dataname = args.dataset
    if dataname == "actor":
        dataname = "film"  # alias

    if dataname == "film":
        return load_geom_gcn_dataset(dataname)
    if dataname in ("chameleon", "squirrel"):
        return load_wiki_new(dataname, no_feat_norm=getattr(args, "no_feat_norm", False))

    raise ValueError(f"Invalid dataname: {args.dataset}")


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.tensor(sparse_mx.data, dtype=torch.float32)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_geom_gcn_dataset(name):
    """
    Loads 'film' (Actor) from geom-gcn formatted files.
    """
    graph_edges_path = os.path.join(DATAPATH, f"geom-gcn/{name}/out1_graph_edges.txt")
    node_feat_label_path = os.path.join(DATAPATH, f"geom-gcn/{name}/out1_node_feature_label.txt")

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name != "film":
        raise ValueError(f"Invalid geom-gcn dataname: {name}")

    with open(node_feat_label_path) as f:
        f.readline()
        for line in f:
            node_id, feat_str, label_str = line.rstrip().split("\t")
            node_id = int(node_id)
            label = int(label_str)

            feature = np.zeros(932, dtype=np.uint8)
            if feat_str:
                feature[np.array(feat_str.split(","), dtype=np.uint16)] = 1

            graph_node_features_dict[node_id] = feature
            graph_labels_dict[node_id] = label

    with open(graph_edges_path) as f:
        f.readline()
        for line in f:
            src, dst = line.rstrip().split("\t")
            src, dst = int(src), int(dst)

            if src not in G:
                G.add_node(src, features=graph_node_features_dict[src], label=graph_labels_dict[src])
            if dst not in G:
                G.add_node(dst, features=graph_node_features_dict[dst], label=graph_labels_dict[dst])

            G.add_edge(src, dst)

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sp.coo_matrix(adj)
    adj = (adj + sp.eye(adj.shape[0])).tocoo().astype(np.float32)  # add self-loops

    features = np.array([feat for _, feat in sorted(G.nodes(data="features"), key=lambda x: x[0])])
    labels = np.array([lab for _, lab in sorted(G.nodes(data="label"), key=lambda x: x[0])], dtype=np.int64)

    # Row-normalize features (as in many geom-gcn loaders)
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum == 0) * 1 + rowsum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    features = sp.diags(r_inv).dot(features)

    edge_index = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long).view(-1, 1)  # [N,1]

    dataset = NCDataset(name)
    dataset.graph = {
        "edge_index": edge_index,
        "node_feat": node_feat,
        "edge_feat": None,
        "num_nodes": node_feat.shape[0],
    }
    dataset.label = y
    return dataset


def load_wiki_new(name, no_feat_norm=False):
    """
    Loads chameleon/squirrel from: data/wiki_new/<name>/<name>_filtered.npz
    """
    npz_path = os.path.join(DATAPATH, f"wiki_new/{name}/{name}_filtered.npz")
    data = np.load(npz_path)

    node_feat = data["node_features"]
    labels = data["node_labels"]
    edges = data["edges"]  # (E,2)

    if not no_feat_norm:
        node_feat = normalize_feat(node_feat)

    edge_index = torch.as_tensor(edges.T, dtype=torch.long)
    node_feat = torch.as_tensor(node_feat, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long).view(-1, 1)  # [N,1]

    dataset = NCDataset(name)
    dataset.graph = {
        "edge_index": edge_index,
        "node_feat": node_feat,
        "edge_feat": None,
        "num_nodes": node_feat.shape[0],
    }
    dataset.label = y
    return dataset