# ---------------------------------------
# DRIVER: Sensitivity Analysis (Presentation)
# ---------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Import your model and sensitivity module
from melanoma_model import params, regimens
from sensitivity_analysis import run_sensitivity
        
# ----------------------------
# Choose regimen
# ----------------------------
# Use whichever regimens your model defines — adjust key if needed
regimen_name = "Default_Regimen"  # try "Combo", "PROTAC_only", etc.
regimen_dict = regimens

folder = "results_sensitivty_presentation"
os.makedirs(folder, exist_ok=True)

# ----------------------------
# Run sensitivity analysis
# ----------------------------
vary_keys = [
    'k_ras_act', 'k_ras_deact',
    'k_raf_act', 'k_raf_deact',
    'k_mek_act', 'k_mek_deact',
    'k_act', 'k_deact',
    'k_rtk_syn', 'k_rtk_deg',
    'k_dusp_prod', 'k_dusp_decay'
]

df_sens = run_sensitivity(
    params=params,
    regimen_dict=regimen_dict,
    param_subset=vary_keys,
    perturbation=0.1,
    t_span=(0, 120)
)

# Save results to CSV
df_sens.to_csv(os.path.join(folder, "results_sensitivity.csv"), index=False)
print("✅ Sensitivity analysis complete. Results saved to results_sensitivity.csv")

# ----------------------------
# Visualization (presentation style)
# ----------------------------

# Sort by absolute sensitivity
df_sens["Abs_Sensitivity"] = np.abs(df_sens["Sensitivity"])
df_sens_sorted = df_sens.sort_values("Abs_Sensitivity", ascending=False)

# Plot
plt.figure(figsize=(8,5))
colors = ['blue' if s < 0 else 'red' for s in df_sens_sorted["Sensitivity"]]
plt.barh(df_sens_sorted["Parameter"], df_sens_sorted["Sensitivity"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Normalized Sensitivity (ΔPI / ΔParameter)")

# Literal result as title (top parameter)
top_param = df_sens_sorted.iloc[0]
plt.title(f"Perturbing {top_param['Parameter']} by ±10% changes PI by {top_param['Sensitivity']:.2f} (largest effect)")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(folder, "figure_sensitivity_presentation_final.png"), dpi=300)

# ----------------------------
# Optional: print top parameters
# ----------------------------
top_params = df_sens.iloc[:5][["Parameter", "Sensitivity"]]
print("\nTop 5 most sensitive parameters:")
print(top_params.to_string(index=False))
