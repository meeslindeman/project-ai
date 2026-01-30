import numpy as np
import matplotlib.pyplot as plt

# Dimensions
dims = np.array([8, 16, 32, 64, 128])

dataset = "disease"

# Mean (%) and std (%)
airport = {
    "Euclidean": {
        "mean": np.array([70.29, 73.26, 80.34, 82.56, 83.00]),
        "std":  np.array([3.17, 2.88, 2.58, 1.54, 1.70]),
    },
    "Projection": {  # euclidean_lorentz
        "mean": np.array([71.76, 77.86, 80.53, 81.20, 82.16]),
        "std":  np.array([2.06, 2.55, 2.91, 2.06, 2.30]),
    },
    "Chen": {  # hypformer
        "mean": np.array([73.32, 83.70, 86.79, 87.84, 87.50]),
        "std":  np.array([3.79, 2.17, 1.56, 1.65, 1.74]),
    },
    "Ours": {  # personal
        "mean": np.array([71.22, 84.92, 88.09, 90.74, 91.38]),
        "std":  np.array([2.84, 2.55, 1.63, 2.00, 1.32]),
    },
}

disease = {
    "Euclidean": {
        "mean": np.array([81.69, 81.10, 82.64, 85.59, 85.91]),
        "std":  np.array([2.83, 2.98, 3.34, 2.79, 3.08]),
    },
    "Projection": {  # euclidean_lorentz
        "mean": np.array([84.69, 87.64, 86.10, 85.31, 85.12]),
        "std":  np.array([3.12, 2.78, 3.19, 3.33, 3.00]),
    },
    "Chen": {  # hypformer
        "mean": np.array([84.64, 88.58, 87.32, 87.72, 88.90]),
        "std":  np.array([2.20, 2.82, 4.33, 3.42, 2.96]),
    },
    "Ours": {  # personal
        "mean": np.array([84.09, 87.16, 87.91, 88.47, 87.72]),
        "std":  np.array([2.71, 3.58, 3.03, 3.31, 3.04]),
    },
}

colors = plt.get_cmap("Set1").colors  # standard matplotlib palette

plt.rcParams.update({
    "font.size": 14,          # base
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

fig, ax = plt.subplots(figsize=(8, 5))

data = airport if dataset == "airport" else disease

for i, (name, d) in enumerate(data.items()):
    mean, std = d["mean"], d["std"]
    color = colors[i]

    # Line + pointwise std bars
    ax.errorbar(
        dims,
        mean,
        yerr=std,
        marker="o",
        linewidth=2,
        capsize=4,
        color=color,
        label=name,
    )

# Formatting
ax.set_xscale("log", base=2)
ax.set_xticks(dims)
ax.set_xticklabels([str(x) for x in dims])
ax.set_xlabel("Dimension")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(65, 95)
ax.grid(True, linestyle="--", alpha=0.35)

# Legend tight and directly above axes
ax.legend(
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    frameon=False,
    handlelength=2.0,
    columnspacing=1.2,
)

fig.tight_layout()

output_file = f"dim_{dataset}.pdf"
fig.savefig(output_file, bbox_inches="tight", format="pdf")
plt.close(fig)
