import os
import re
import numpy as np

LOG_ROOT = "logs"   # root directory containing dataset folders

BEST_TEST_RE = re.compile(r"corresponding test ([0-9.]+)")

results = {}  


for dataset in os.listdir(LOG_ROOT):
    dataset_path = os.path.join(LOG_ROOT, dataset)
    if not os.path.isdir(dataset_path):
        continue

    results[dataset] = {}

    for fname in os.listdir(dataset_path):
        if not fname.startswith("curv_") or not fname.endswith(".log"):
            continue

        curvature = fname.replace("curv_", "").replace(".log", "")
        fpath = os.path.join(dataset_path, fname)

        with open(fpath, "r") as f:
            content = f.read()

        # Extract all "corresponding test X" values
        tests = [float(x) for x in BEST_TEST_RE.findall(content)]

        if len(tests) == 0:
            print(f"Warning: no test scores found in {fpath}")
            continue

        results[dataset][curvature] = tests


print("\n=== CURVATURE ABLATION RESULTS (mean ± std in %) ===\n")

for dataset, curvs in results.items():
    print(f"Dataset: {dataset}")
    for curv, scores in sorted(curvs.items(), key=lambda x: float(x[0])):
        arr = np.array(scores) * 100  # convert to %
        mean = arr.mean()
        std = arr.std()
        print(f"  curvature {curv}: {mean:.2f}% ± {std:.2f}%  (n={len(arr)})")
    print()
