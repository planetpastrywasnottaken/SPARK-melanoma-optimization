# run_optimization.py (cluster-aware optimization)

import os
import pandas as pd
import numpy as np
from melanoma_model import simulate, default_y0

# ----------------------------
# Output folders
# ----------------------------
out_dir = "results_cluster_aware_optimization"
ts_dir = os.path.join(out_dir, "time_series")
os.makedirs(ts_dir, exist_ok=True)

# ----------------------------
# Cluster definitions
# ----------------------------
patient_clusters = {
    "Proliferative": {
        "params": {
            "MAPKpp": [0.45, 0.65],
            "AKT": [0.1, 0.3],
            "Pers_conv": [0.01, 0.03]
        }
    },
    "Invasive": {
        "params": {
            "AKT": [0.4, 0.7],
            "Pers_conv": [0.03, 0.07]
        }
    },
    "Persister": {
        "params": {
            "Init_pers_frac": [0.05, 0.15],
            "Pers_conv": [0.05, 0.1],
            "Reversion": [0.01, 0.03]
        }
    },
    "RTK_Adaptive": {
        "params": {
            "RTK_expr": [0.6, 1.0],
            "MAPKpp": [0.3, 0.5],
            "Pers_conv": [0.04, 0.08]
        }
    }
}

# ----------------------------
# Sampling
# ----------------------------
def sample_patient(cluster_params):
    return {
        k: np.random.uniform(v[0], v[1]) if isinstance(v, list) else v
        for k, v in cluster_params.items()
    }

# ----------------------------
# Regimen generator
# ----------------------------
def generate_regimens(n=30):
    regs = []
    for i in range(n):
        regs.append({
            "protac": {
                "dose": np.random.uniform(0.5, 2.5),
                "start": np.random.choice([0, 6]),
                "interval": np.random.choice([12, 24]),
                "n_pulses": np.random.choice([2, 3, 4, 5]),
                "half_life": np.random.uniform(3, 9)
            },
            "rtk": {
                "dose": np.random.uniform(0.5, 2.5),
                "start": np.random.choice([0, 6]),
                "interval": np.random.choice([12, 24]),
                "n_pulses": np.random.choice([2, 3, 4, 5]),
                "half_life": np.random.uniform(6, 16)
            },
            "name": f"Reg_{i+1}"
        })
    return regs

regimens = generate_regimens(30)

# ----------------------------
# Cluster-specific objective
# ----------------------------
def cluster_objective(cluster, result):
    ts = result["time_series"]
    t = ts["t"].values
    pers_frac = ts["N_pers"] / (ts["N_bulk"] + ts["N_pers"] + 1e-12)

    early = t < 40
    late = t > 80

    PI_early = np.trapezoid(pers_frac[early], t[early]) / 40
    PI_late = np.trapezoid(pers_frac[late], t[late]) / 40
    PI_total = result["PI_timeavg"]

    harm = result["harm_index"]

    if cluster == "Proliferative":
        score = (
            1 - PI_total
            - 0.4 * PI_late
            - 0.2 * harm
        )

    elif cluster == "Invasive":
        score = (
            1 - PI_late
            - 0.3 * PI_total
            - 0.2 * harm
        )

    elif cluster == "Persister":
        score = (
            1 - PI_early
            - 0.4 * PI_total
            - 0.1 * harm
        )

    elif cluster == "RTK_Adaptive":
        rebound = np.max(ts["MAPKpp"].values[-50:])
        score = (
            1 - rebound
            - 0.3 * PI_late
            - 0.2 * harm
        )

    else:
        score = 1 - PI_total - harm

    return score

# ----------------------------
# Run optimization
# ----------------------------
results = []
N_patients = 20

for cluster, info in patient_clusters.items():
    print(f"\nOptimizing for cluster: {cluster}")

    for pid in range(N_patients):
        patient = sample_patient(info["params"])

        for reg in regimens:
            res = simulate(reg, patient)
            res["cluster"] = cluster
            res["patient_id"] = pid
            res["cluster_score"] = cluster_objective(cluster, res)

            results.append(res)

            ts_file = f"{ts_dir}/ts_{cluster}_{pid}_{reg['name']}.csv"
            res["time_series"].to_csv(ts_file, index=False)

# ----------------------------
# Summarize + select best
# ----------------------------
df = pd.DataFrame(results)

summary = (
    df.groupby(["cluster", "regimen"])
    .agg(cluster_score=("cluster_score", "mean"))
    .reset_index()
)

best = summary.loc[summary.groupby("cluster")["cluster_score"].idxmax()]

summary.to_csv(f"{out_dir}/cluster_summary.csv", index=False)
best.to_csv(f"{out_dir}/best_regimens_per_cluster.csv", index=False)

print("\nBest regimen per cluster:")
print(best)
