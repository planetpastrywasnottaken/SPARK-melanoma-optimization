# plot_cluster_optimization.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
SUMMARY_CSV = os.path.join(BASE_DIR, "cluster_summary.csv")
BEST_CSV = os.path.join(BASE_DIR, "best_regimens_per_cluster.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(SUMMARY_CSV)
best = pd.read_csv(BEST_CSV)

clusters = df["cluster"].unique()
regimens = df["regimen"].unique()

# ----------------------------
# 1. Heatmap: Regimen × Cluster
# ----------------------------
pivot = df.pivot(index="regimen", columns="cluster", values="cluster_score")

plt.figure(figsize=(10, 8))
im = plt.imshow(pivot.values, aspect="auto")

plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
plt.yticks(range(len(pivot.index)), pivot.index)
plt.colorbar(im, label="Mean Cluster Score")

plt.title("Cluster-Aware Regimen Performance Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "heatmap_regimen_vs_cluster.png"), dpi=300)
plt.close()

# ----------------------------
# 2. Bar chart: Best regimen per cluster
# ----------------------------
plt.figure(figsize=(8, 6))
plt.bar(best["cluster"], best["cluster_score"])

plt.ylabel("Best Mean Cluster Score")
plt.title("Optimal Regimen per Patient Cluster")
plt.xticks(rotation=25, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "best_regimen_per_cluster.png"), dpi=300)
plt.close()

# ----------------------------
# 3. Global-best vs Cluster-best
# ----------------------------
# Global-best regimen = highest average score across all clusters
global_scores = df.groupby("regimen")["cluster_score"].mean()
global_best_regimen = global_scores.idxmax()

comparison = []
for c in clusters:
    cluster_best = best.loc[best["cluster"] == c, "cluster_score"].values[0]
    global_score = df[
        (df["cluster"] == c) & (df["regimen"] == global_best_regimen)
    ]["cluster_score"].values[0]

    comparison.append([c, cluster_best, global_score])

comp_df = pd.DataFrame(
    comparison, columns=["cluster", "cluster_best", "global_best"]
)

x = np.arange(len(clusters))
width = 0.35

plt.figure(figsize=(9, 6))
plt.bar(x - width/2, comp_df["cluster_best"], width, label="Cluster-Optimal")
plt.bar(x + width/2, comp_df["global_best"], width, label="Global-Optimal")

plt.xticks(x, comp_df["cluster"], rotation=25, ha="right")
plt.ylabel("Mean Cluster Score")
plt.title("Cluster-Optimal vs Global-Optimal Regimens")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_vs_global_optimal.png"), dpi=300)
plt.close()

# ----------------------------
# 4. Regimen robustness plot (top 5 regimens)
# ----------------------------
top_regimens = (
    df.groupby("regimen")["cluster_score"]
    .mean()
    .sort_values(ascending=False)
    .head(5)
    .index
)

plt.figure(figsize=(9, 6))
for reg in top_regimens:
    vals = df[df["regimen"] == reg].sort_values("cluster")["cluster_score"]
    plt.plot(clusters, vals, marker="o", label=reg)

plt.ylabel("Mean Cluster Score")
plt.title("Regimen Robustness Across Clusters")
plt.legend()
plt.xticks(rotation=25, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "regimen_robustness.png"), dpi=300)
plt.close()

# ----------------------------
# 5. Score distribution per cluster
# ----------------------------
plt.figure(figsize=(9, 6))
data = [
    df[df["cluster"] == c]["cluster_score"].values
    for c in clusters
]

plt.boxplot(data, labels=clusters)
plt.ylabel("Cluster Score")
plt.title("Distribution of Regimen Performance per Cluster")
plt.xticks(rotation=25, ha="right")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cluster_score_distributions.png"), dpi=300)
plt.close()

print("All figures saved to:", FIG_DIR)
