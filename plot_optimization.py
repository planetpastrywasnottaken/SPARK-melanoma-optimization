import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load optimization results
# ----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
opt_csv = os.path.join(script_dir, 'results_optimization_presentation', 'optimization_test_results.csv')
opt_csv = os.path.normpath(opt_csv)
opt = pd.read_csv(opt_csv)
print("CSV loaded from:", opt_csv)

# ----------------------------
# Ensure consistent column names
# ----------------------------
# Rename columns from run_optimization output to match plotting expectations
if 'COS' in opt.columns:
    opt.rename(columns={'COS':'Composite_Score'}, inplace=True)
if 'harm_index' in opt.columns:
    opt.rename(columns={'harm_index':'Harm_Index'}, inplace=True)

# Convert key columns to numeric (in case CSV read them as object)
numeric_cols = ['Composite_Score','AUC_PROT','AUC_RTK','CMAX_PROT','CMAX_RTK','Harm_Index']
for col in numeric_cols:
    if col in opt.columns:
        opt[col] = pd.to_numeric(opt[col], errors='coerce')

# ----------------------------
# Paths
# ----------------------------
output_dir = os.path.join(script_dir, "results_optimization_presentation")
fig_dir = os.path.join(output_dir, 'figures_optimization')
os.makedirs(fig_dir, exist_ok=True)

# ----------------------------
# 1️⃣ Heatmap: Composite Score by cluster vs regimen
# ----------------------------
agg = opt.groupby(['cluster','regimen']).agg({'Composite_Score':'mean'}).reset_index()
heat = agg.pivot(index='cluster', columns='regimen', values='Composite_Score').fillna(0)

# Order clusters and columns by performance
cluster_order = heat.max(axis=1).sort_values(ascending=False).index.tolist()
col_order = heat.mean().sort_values(ascending=False).index.tolist()
heat = heat.reindex(cluster_order)[col_order]

plt.figure(figsize=(12, max(3, 0.5*heat.shape[0])))
sns.heatmap(heat, annot=True, fmt=".2f", cmap='viridis', cbar_kws={'label':'Composite Score'})
plt.title("Cluster-specific Optimization: Composite Score")
plt.ylabel("Cluster")
plt.xlabel("Regimen")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "heatmap_cluster_regimen_composite.png"), dpi=300)
plt.close()

# ----------------------------
# 2️⃣ Feature difference bars (top 10% vs others)
# ----------------------------
top_frac = 0.10
top_by_cluster = opt.groupby('cluster').apply(
    lambda df: df[df['Composite_Score'] <= df['Composite_Score'].quantile(top_frac)]
).reset_index(drop=True)

feature_cols = [c for c in ['AUC_PROT','AUC_RTK','CMAX_PROT','CMAX_RTK','Harm_Index'] if c in opt.columns]

diff_frames = []
for cluster, group in opt.groupby('cluster'):
    top = top_by_cluster[top_by_cluster['cluster']==cluster]
    top_mean = top[feature_cols].mean()
    other = opt[(opt['cluster']==cluster) & (~opt.index.isin(top.index))]
    other_mean = other[feature_cols].mean()
    diff = (top_mean - other_mean).rename(cluster)
    diff_frames.append(diff)

diff_df = pd.concat(diff_frames, axis=1).T.fillna(0)

for cluster in diff_df.index:
    plt.figure(figsize=(6,3.5))
    vals = diff_df.loc[cluster].sort_values()
    vals.plot(kind='barh', color='skyblue')
    plt.axvline(0, color='k', linewidth=0.7)
    plt.title(f"Feature differences (top {int(top_frac*100)}%) vs others — {cluster}")
    plt.xlabel("Top - Others (units depend on feature)")
    plt.tight_layout()
    cluster_safe = cluster.replace(' ', '_').replace('/', '_').replace('\\','_')
    plt.savefig(os.path.join(fig_dir, f"feature_diff_{cluster_safe}.png"), dpi=300)
    plt.close()

# ----------------------------
# 3️⃣ Line plot: Composite Score vs Regimen
# ----------------------------
plt.figure(figsize=(10,5))
for cluster in opt['cluster'].unique():
    cluster_data = opt[opt['cluster']==cluster].groupby('regimen')['Composite_Score'].mean()
    plt.plot(cluster_data.index, cluster_data.values, marker='o', label=cluster)
plt.xlabel("Regimen")
plt.ylabel("Composite Score")
plt.title("Composite Score across Regimens per Cluster")
plt.xticks(rotation=45)
plt.legend(title="Cluster", bbox_to_anchor=(1.05,1))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "line_composite_score_vs_regimen.png"), dpi=300)
plt.close()

print("All three optimization figures complete.")
