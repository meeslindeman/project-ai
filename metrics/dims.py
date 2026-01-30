import os
import re
import numpy as np

LOG_ROOT = "logs"
BEST_TEST_RE = re.compile(r"corresponding test ([0-9.]+)")

results = {}

for dataset in os.listdir(LOG_ROOT):
    dataset_path = os.path.join(LOG_ROOT, dataset)
    if not os.path.isdir(dataset_path):
        continue

    results[dataset] = {}

    for model in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model)
        if not os.path.isdir(model_path):
            continue

        results[dataset][model] = {}

        for fname in os.listdir(model_path):
            if not fname.startswith("dim_") or not fname.endswith(".log"):
                continue

            dim = fname.replace("dim_", "").replace(".log", "")
            fpath = os.path.join(model_path, fname)

            with open(fpath) as f:
                tests = [float(x) for x in BEST_TEST_RE.findall(f.read())]

            if tests:
                results[dataset][model][dim] = tests


print("\n=== DIMENSION ABLATION RESULTS (mean ± std in %) ===\n")

for dataset, models in results.items():
    print(f"Dataset: {dataset}")
    for model, dims in models.items():
        print(f"  Model: {model}")
        for dim, scores in sorted(dims.items(), key=lambda x: int(x[0])):
            arr = np.array(scores) * 100
            mean = arr.mean()
            std = arr.std(ddof=1)
            print(f"    dim {dim}: {mean:.2f}% ± {std:.2f}%  (n={len(arr)})")
    print()
