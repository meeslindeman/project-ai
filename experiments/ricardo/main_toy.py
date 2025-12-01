import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict

from manifolds.personal.layer import LorentzFC, LorentzMLR
from manifolds.personal.lorentz import Lorentz

SEEDS = range(3)
HYPERBOLIC_DISTANCES = range(1, 80, 3)
DEPTHS = [1, 2, 4, 8]         
LEARNING_RATE = 0.001
MAX_ITERATIONS = 10000
LOSS_THRESHOLD = 0.01
BATCH_SIZE = 64
DIM = 4
OUTPUT_FILENAME = ".experiments/ricardo/hyperplane_distance_results.txt"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tangent_space_mse_loss(y_pred: torch.Tensor, y_target: torch.Tensor, manifold: Lorentz) -> torch.Tensor:
    tangent_pred = manifold.logmap0(y_pred)
    tangent_target = manifold.logmap0(y_target)
    return nn.functional.mse_loss(tangent_pred, tangent_target)


def create_target_point(hyperbolic_distance: int, manifold: Lorentz, dim: int) -> torch.Tensor:
    d = float(hyperbolic_distance)
    sqrt_k = manifold.k().sqrt().double()

    time = (1.0 / sqrt_k) * torch.cosh(sqrt_k * d)
    space_first = (1.0 / sqrt_k) * torch.sinh(sqrt_k * d)

    point = torch.zeros(dim + 1, dtype=torch.float64)
    point[0] = time
    point[1] = space_first
    return point

class StackedLorentzFC(nn.Module):
    """
    Applies `depth` many LorentzFC layers in sequence.

    Each LorentzFC has in_features = out_features = DIM,
    so the spatial dimension stays constant and the Lorentz time+space size is DIM+1.
    """ 
    def __init__(self, dim: int, manifold: Lorentz, depth: int, activation=nn.Identity()):
        super().__init__()
        self.manifold = manifold
        self.dim = dim
        self.depth = depth

        layers = []
        for _ in range(depth):
            layers.append(
                LorentzFC(
                    in_features=dim,
                    out_features=dim,
                    manifold=manifold,
                    reset_params="eye",
                    a_default=0.01,
                    activation=activation,  
                ).double()
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def run_experiment(hyperbolic_distance: int, depth: int, seed: int, manifold: Lorentz, dim: int) -> int:
    set_seed(seed)

    # Model: stack of depth many LorentzFC layers
    model = StackedLorentzFC(
        dim=dim,
        manifold=manifold,
        depth=depth,
        activation=nn.Identity()
    )
    
    # Create input point at some fixed distance from origin
    dist_input = np.sqrt(2)
    sqrt_k = manifold.k().sqrt()
    time_input = (1.0 / sqrt_k) * torch.cosh(sqrt_k * dist_input)
    space_input = (1.0 / sqrt_k) * torch.sinh(sqrt_k * dist_input)

    x = torch.tensor(
        [time_input.item(), space_input.item()] + [0.0] * (dim - 1),
        dtype=torch.float64,
    )
    x = x.unsqueeze(0).repeat(BATCH_SIZE, 1)

    # Create target point at specified hyperbolic distance
    y = create_target_point(hyperbolic_distance, manifold, dim)
    y = y.unsqueeze(0).repeat(BATCH_SIZE, 1)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for i in range(MAX_ITERATIONS):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = tangent_space_mse_loss(y_pred, y, manifold)

        if loss.item() < LOSS_THRESHOLD:
            return i + 1

        loss.backward()

        total_param_norm = 0.0
        for _, p in model.named_parameters():
            if p.grad is not None:
                total_param_norm += p.norm().item() ** 2
        total_param_norm = np.sqrt(total_param_norm)
        scale_factor = max(1.0, total_param_norm)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale_factor)

        optimizer.step()

        if i % 1000 == 0:
            print(
                f"  [depth={depth}] step {i:5d}: dist={hyperbolic_distance}, "
                f"loss={loss.item():.6f}, pred_time={y_pred[0,0].item():.6f}, "
                f"target_time={y[0,0].item():.6f}"
            )

    return MAX_ITERATIONS


def save_results_to_txt(results: Dict[int, Dict[int, List[int]]], filename: str):
    """
    results[depth][dist] = list of iterations over seeds
    """
    with open(filename, 'w') as f:
        f.write("Hyperplane Distance Convergence Results with Depth Scaling\n")
        f.write(f"Batch Size: {BATCH_SIZE}, Seeds: {len(SEEDS)}, Max Iterations: {MAX_ITERATIONS}\n\n")

        for depth in sorted(results.keys()):
            header1 = f"[Depth = {depth}] {'Distance':<10} | {'Mean ± Std (iters)':<25} | {'Status':<10}"
            f.write("=" * len(header1) + "\n")
            f.write(header1 + "\n")
            f.write("-" * len(header1) + "\n")

            for dist, iters in sorted(results[depth].items()):
                res = np.array(iters)
                mean_iters = np.mean(res)
                std_iters = np.std(res)

                if mean_iters >= MAX_ITERATIONS:
                    status = "FAILED"
                elif mean_iters > MAX_ITERATIONS * 0.8:
                    status = "STRUGGLE"
                else:
                    status = "OK"

                f.write(f"{dist:<10} | {mean_iters:7.1f} ± {std_iters:5.1f}    | {status:<10}\n")

            f.write("\n")
    print(f"\nResults saved to '{filename}'")


def main():
    print("🚀 Depth-scaling Hyperplane Distance Test (no ReLU)")
    print(f"Distances: {min(HYPERBOLIC_DISTANCES)}–{max(HYPERBOLIC_DISTANCES)}, depths: {DEPTHS}")
    print(f"Seeds: {len(SEEDS)}, Batch Size: {BATCH_SIZE}, Dimension: {DIM}")
    print("-" * 80)

    manifold = Lorentz(k=1)
    print(f"Manifold curvature k = {manifold.k().item():.4f}")
    print("-" * 80)

    # results[depth][dist] = [iters_per_seed]
    results: Dict[int, Dict[int, List[int]]] = {
        depth: {d: [] for d in HYPERBOLIC_DISTANCES} for depth in DEPTHS
    }

    for depth in DEPTHS:
        print(f"\n=== Testing depth = {depth} ===")
        convergence_failed = False
        first_failure_distance = None

        for dist in HYPERBOLIC_DISTANCES:
            print(f"\n📏 depth={depth}, hyperbolic distance={dist} (e^{dist} ≈ {np.exp(dist):.2e})")

            if convergence_failed:
                print(f"   Skipping (failed already at distance {first_failure_distance})")
                results[depth][dist] = [MAX_ITERATIONS] * len(SEEDS)
                continue

            current_iterations = []
            for seed in SEEDS:
                iters = run_experiment(
                    hyperbolic_distance=dist,
                    depth=depth,
                    seed=seed,
                    manifold=manifold,
                    dim=DIM,
                )
                current_iterations.append(iters)

            results[depth][dist] = current_iterations
            mean_iters = np.mean(current_iterations)

            if mean_iters >= MAX_ITERATIONS:
                convergence_failed = True
                first_failure_distance = dist
                print(f"   ❌ FAILED (mean iters={mean_iters:.1f}), will skip larger distances.")
            elif mean_iters > MAX_ITERATIONS * 0.8:
                print(f"   ⚠️ STRUGGLING (mean iters={mean_iters:.1f})")
            else:
                print(f"   ✅ OK (mean iters={mean_iters:.1f})")

    # Summary to stdout
    print("\n" + "=" * 80)
    print("📊 Convergence Summary by Depth")
    print("=" * 80)

    for depth in DEPTHS:
        print(f"\n[Depth = {depth}]")
        header1 = f"{'Distance':<10} | {'Mean ± Std (iters)':<25} | {'Status':<10}"
        print(header1)
        print("-" * len(header1))

        for dist, iters in sorted(results[depth].items()):
            res = np.array(iters)
            mean_iters = np.mean(res)
            std_iters = np.std(res)

            if mean_iters >= MAX_ITERATIONS:
                status = "FAILED"
            elif mean_iters > MAX_ITERATIONS * 0.8:
                status = "STRUGGLE"
            else:
                status = "OK"

            print(f"{dist:<10} | {mean_iters:7.1f} ± {std_iters:5.1f}    | {status:<10}")

    save_results_to_txt(results, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
