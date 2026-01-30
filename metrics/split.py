import os
import re
import numpy as np

LOG_ROOT = "logs"

BEST_TEST_RE = re.compile(r"corresponding test ([0-9.]+)")

DATASETS = ["squirrel", "chameleon", "actor", "disease", "airport"]
FUSIONS = ["concat_direct", "concat_logradius"]

results = {} 

for dataset in DATASETS:
    results[dataset] = {}

    for fusion in FUSIONS:
        log_path = os.path.join(LOG_ROOT, dataset, fusion, "splits.log")

        if not os.path.exists(log_path):
            print(f"Missing log file: {log_path}")
            continue

        with open(log_path, "r") as f:
            content = f.read()

        tests = [float(x) for x in BEST_TEST_RE.findall(content)]

        if len(tests) == 0:
            print(f"No test scores found in {log_path}")
            continue

        results[dataset][fusion] = tests


print("\n=== HEAD FUSION RESULTS (mean ± std in %) ===\n")

for dataset, fusion_dict in results.items():
    print(f"Dataset: {dataset}")

    for fusion, scores in fusion_dict.items():
        arr = np.array(scores) * 100  # convert to %
        mean = arr.mean()
        std = arr.std()
        print(f"  {fusion:15s}: {mean:.2f}% ± {std:.2f}%  (n={len(arr)})")

    print()
