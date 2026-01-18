import numpy as np
import pandas as pd
from melanoma_model import simulate, summarize

def run_sensitivity(params, regimen_dict, param_subset=None, perturbation=0.1, t_span=(0,120)):
    """
    Perform local sensitivity analysis on selected parameters.
    
    params: dict of model parameters
    regimen_dict: regimen setup (PROTAC, RTKi, etc.)
    param_subset: list of parameters to test (if None, uses all)
    perturbation: fractional change to apply (e.g., 0.1 = ±10%)
    t_span: simulation time range
    
    Returns: pandas DataFrame with sensitivities
    """
    if param_subset is None:
        param_subset = list(params.keys())

    baseline_sol = simulate(regimen_dict, params, t_span=t_span)
    baseline_metrics = summarize(baseline_sol, regimen_name="baseline", dose_sum=0)
    baseline_PI = baseline_metrics["PI_final"]

    results = []

    for p in param_subset:
        # make copies for +/- perturbation
        params_plus = params.copy()
        params_minus = params.copy()

        params_plus[p] = params[p] * (1 + perturbation)
        params_minus[p] = params[p] * (1 - perturbation)

        # run both simulations
        sol_plus = simulate(regimen_dict, params_plus, t_span=t_span)
        sol_minus = simulate(regimen_dict, params_minus, t_span=t_span)

        PI_plus = summarize(sol_plus, regimen_name=f"{p}_plus", dose_sum=0)["PI_final"]
        PI_minus = summarize(sol_minus, regimen_name=f"{p}_minus", dose_sum=0)["PI_final"]

        # finite difference approximation of sensitivity
        S = (PI_plus - PI_minus) / (2 * baseline_PI * perturbation)

        results.append({
            "Parameter": p,
            "Sensitivity": S,
            "PI_plus": PI_plus,
            "PI_minus": PI_minus,
            "Baseline_PI": baseline_PI
        })

    return pd.DataFrame(results)
