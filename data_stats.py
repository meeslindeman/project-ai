import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

import networkx as nx

from dataset import load_nc_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def edge_index_to_undirected_nx(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    """
    Converts PyG-style edge_index [2, E] to an undirected NetworkX graph.
    Removes self-loops and parallel edges automatically (Graph is simple).
    """
    ei = edge_index.detach().cpu()
    if ei.dim() != 2 or ei.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(ei.shape)}")

    src = ei[0].tolist()
    dst = ei[1].tolist()

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = [(u, v) for u, v in zip(src, dst) if u != v]
    G.add_edges_from(edges)
    return G


def approx_diameter(G: nx.Graph, num_sources: int = 16, seed: int = 0) -> int:
    """
    Approximate (graph) diameter by running BFS from a few sources and taking max ecc.
    Exact diameter can be expensive on larger graphs; this is typically good enough for comparison.
    """
    rng = random.Random(seed)

    nodes = list(G.nodes())
    if not nodes:
        return 0

    # bias sources toward higher-degree nodes + a few random
    deg_sorted = sorted(G.degree, key=lambda x: -x[1])
    top = [n for n, _ in deg_sorted[: max(1, num_sources // 2)]]
    rest = [rng.choice(nodes) for _ in range(max(0, num_sources - len(top)))]
    sources = list(dict.fromkeys(top + rest))  # unique, preserve order

    best = 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(G, s)
        if lengths:
            best = max(best, max(lengths.values()))
    return int(best)


def effective_diameter_from_samples(
    G: nx.Graph,
    num_pairs: int = 5000,
    quantile: float = 0.9,
    seed: int = 0,
) -> float:
    """
    Estimate effective diameter: q-quantile of shortest-path distances over sampled node pairs
    (within the same connected component).
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return 0.0

    # precompute connected components for fast “same component” sampling
    comps = [list(c) for c in nx.connected_components(G)]
    comps = [c for c in comps if len(c) >= 2]
    if not comps:
        return 0.0

    # sample pairs by sampling a component proportional to size
    sizes = np.array([len(c) for c in comps], dtype=np.float64)
    probs = sizes / sizes.sum()

    dists: List[int] = []
    for _ in range(num_pairs):
        ci = int(np.searchsorted(np.cumsum(probs), rng.random(), side="right"))
        c = comps[min(ci, len(comps) - 1)]
        u, v = rng.sample(c, 2)
        try:
            d = nx.shortest_path_length(G, u, v)
            dists.append(int(d))
        except nx.NetworkXNoPath:
            # should be rare since u,v from same component, but keep safe
            continue

    if not dists:
        return 0.0
    return float(np.quantile(np.array(dists, dtype=np.float64), quantile))


def pick_roots(G: nx.Graph, strategy: str, k: int, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if not nodes:
        return []

    if strategy == "top_degree":
        deg_sorted = sorted(G.degree, key=lambda x: -x[1])
        return [n for n, _ in deg_sorted[:k]]
    elif strategy == "random":
        k = min(k, len(nodes))
        return rng.sample(nodes, k)
    elif strategy == "mixed":
        # half top-degree, half random
        k1 = max(1, k // 2)
        k2 = max(0, k - k1)
        top = pick_roots(G, "top_degree", k1, seed=seed)
        remaining = [n for n in nodes if n not in set(top)]
        if remaining and k2 > 0:
            rnd = rng.sample(remaining, min(k2, len(remaining)))
        else:
            rnd = []
        return top + rnd
    else:
        raise ValueError(f"Unknown root strategy: {strategy}")


def depth_stats_from_roots(G: nx.Graph, roots: List[int]) -> Dict[str, float]:
    """
    Define “node depth” as BFS distance from a root.
    For non-tree graphs, this is depth in the BFS tree from that root.
    We aggregate depths over all roots (concatenate all BFS distances).
    """
    all_depths: List[int] = []
    per_root_max: List[int] = []

    for r in roots:
        lengths = nx.single_source_shortest_path_length(G, r)
        depths = list(lengths.values())
        if not depths:
            continue
        all_depths.extend(depths)
        per_root_max.append(max(depths))

    if not all_depths:
        return {
            "depth_mean": float("nan"),
            "depth_std": float("nan"),
            "depth_p90": float("nan"),
            "depth_max": float("nan"),
            "ecc_mean": float("nan"),
            "ecc_max": float("nan"),
        }

    a = np.array(all_depths, dtype=np.float64)
    ecc = np.array(per_root_max, dtype=np.float64) if per_root_max else np.array([np.nan])

    # sample standard deviation (ddof=1) per your preference
    depth_std = float(a.std(ddof=1)) if a.size >= 2 else 0.0

    return {
        "depth_mean": float(a.mean()),
        "depth_std": depth_std,
        "depth_p90": float(np.quantile(a, 0.9)),
        "depth_max": float(a.max()),
        "ecc_mean": float(np.nanmean(ecc)),
        "ecc_max": float(np.nanmax(ecc)),
    }


def core_stats(G: nx.Graph) -> Dict[str, float]:
    """
    k-core hierarchy (core number). Works on undirected simple graphs.
    """
    if G.number_of_nodes() == 0:
        return {"core_max": 0.0, "core_mean": 0.0, "core_std": 0.0}

    core = nx.core_number(G)
    vals = np.array(list(core.values()), dtype=np.float64)
    core_std = float(vals.std(ddof=1)) if vals.size >= 2 else 0.0
    return {
        "core_max": float(vals.max()),
        "core_mean": float(vals.mean()),
        "core_std": core_std,
    }


def leaf_branch_stats(G: nx.Graph) -> Dict[str, float]:
    """
    Simple tree-likeness proxies.
    """
    deg = np.array([d for _, d in G.degree()], dtype=np.float64)
    if deg.size == 0:
        return {"leaf_frac": 0.0, "avg_deg": 0.0, "avg_deg_gt1": 0.0}

    leaf_frac = float((deg == 1).mean())
    avg_deg = float(deg.mean())
    deg_gt1 = deg[deg > 1]
    avg_deg_gt1 = float(deg_gt1.mean()) if deg_gt1.size > 0 else 0.0

    return {"leaf_frac": leaf_frac, "avg_deg": avg_deg, "avg_deg_gt1": avg_deg_gt1}


def summarize_components(G: nx.Graph) -> Dict[str, float]:
    comps = [len(c) for c in nx.connected_components(G)]
    comps = np.array(comps, dtype=np.float64) if comps else np.array([0.0])
    return {
        "num_components": float(len(comps)),
        "largest_cc_frac": float(comps.max() / max(1.0, G.number_of_nodes())),
    }


def compute_all_stats(
    dataset_name: str,
    dataset,
    args,
) -> Dict[str, float]:
    num_nodes = int(dataset.graph["num_nodes"])
    edge_index = dataset.graph["edge_index"]
    G = edge_index_to_undirected_nx(edge_index, num_nodes)

    stats: Dict[str, float] = {}
    stats["num_nodes"] = float(G.number_of_nodes())
    stats["num_edges_undirected"] = float(G.number_of_edges())

    stats.update(summarize_components(G))
    stats.update(leaf_branch_stats(G))
    stats.update(core_stats(G))

    # Diameter / effective diameter (approx / sampled)
    stats["diameter_approx"] = float(approx_diameter(G, num_sources=args.diam_sources, seed=args.seed))
    stats["eff_diam_q90"] = float(
        effective_diameter_from_samples(
            G,
            num_pairs=args.effdiam_pairs,
            quantile=0.9,
            seed=args.seed,
        )
    )

    # Depth stats (BFS depth from selected roots)
    roots = pick_roots(G, strategy=args.depth_roots, k=args.num_roots, seed=args.seed)
    ds = depth_stats_from_roots(G, roots)
    stats.update(ds)

    return stats


def print_stats(name: str, stats: Dict[str, float]) -> None:
    keys_order = [
        "num_nodes",
        "num_edges_undirected",
        "num_components",
        "largest_cc_frac",
        "avg_deg",
        "leaf_frac",
        "core_max",
        "core_mean",
        "diameter_approx",
        "eff_diam_q90",
        "depth_mean",
        "depth_std",
        "depth_p90",
        "depth_max",
        "ecc_mean",
        "ecc_max",
    ]
    print(f"\n=== {name} ===")
    for k in keys_order:
        if k in stats:
            v = stats[k]
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                print(f"{k:>18s}: {v}")
            else:
                print(f"{k:>18s}: {v:.6g}" if isinstance(v, float) else f"{k:>18s}: {v}")


@dataclass
class ArgsForLoader:
    """
    Minimal args object for load_nc_dataset(args).
    Add fields here if your load_nc_dataset expects more.
    """
    dataset: str
    no_feat_norm: bool
    normalize_feats: bool
    use_feats: bool


def main():
    p = argparse.ArgumentParser("Compute hierarchy/depth stats for your NC datasets")

    p.add_argument("--dataset", type=str, default="chameleon",
                   choices=["chameleon", "squirrel", "actor", "airport", "disease"])
    p.add_argument("--all", action="store_true",
                   help="If set, ignore --dataset and compute for all 5 datasets.")

    # Keep these to match your training script so load_nc_dataset(args) behaves identically
    p.add_argument("--no_feat_norm", action="store_true",
                   help="Don't normalize features (chameleon/squirrel)")
    p.add_argument("--normalize_feats", action="store_true", default=True,
                   help="Normalize features (airport/disease)")
    p.add_argument("--use_feats", action="store_true", default=True,
                   help="Use node features (disease)")

    # Depth / diameter settings
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depth_roots", type=str, default="mixed",
                   choices=["top_degree", "random", "mixed"],
                   help="How to choose BFS roots for node-depth statistics.")
    p.add_argument("--num_roots", type=int, default=8,
                   help="Number of roots used for BFS depth statistics.")
    p.add_argument("--diam_sources", type=int, default=16,
                   help="Number of BFS sources for approximate diameter.")
    p.add_argument("--effdiam_pairs", type=int, default=5000,
                   help="Number of sampled node pairs for effective diameter estimate.")

    args = p.parse_args()
    set_seed(args.seed)

    datasets = ["actor", "squirrel", "chameleon", "airport", "disease"] if args.all else [args.dataset]

    for dname in datasets:
        loader_args = ArgsForLoader(
            dataset=dname,
            no_feat_norm=args.no_feat_norm,
            normalize_feats=args.normalize_feats,
            use_feats=args.use_feats,
        )
        dataset = load_nc_dataset(loader_args)
        stats = compute_all_stats(dname, dataset, args)
        print_stats(dname, stats)


if __name__ == "__main__":
    main()
