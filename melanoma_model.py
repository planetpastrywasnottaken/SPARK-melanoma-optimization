import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------------------------
# Load parameters from CSV
# ----------------------------
df = pd.read_csv("csv/params_literature.csv", header=0, quotechar='"')
params = {}
for _, row in df.iterrows():
    try:
        value = float(row['value'])
    except:
        value = None
    params[row['parameter']] = value

# Add default persister index parameters if missing
params.setdefault('ci_max', 0.6)
params.setdefault('ci_alpha', 0.1)
params.setdefault('ci_b', 1.0)

default_y0 = np.array([
    0.1,  # RTK
    0.0,  # RAS
    0.0,  # RAF
    0.0,  # MEK
    0.2,  # MAPKpp
    0.1,  # AKT_p
    0.0,  # cFos
    0.0,  # CyclinD
    0.9,  # N_bulk
    0.1,  # N_pers
    0.0,  # DUSP
    1.0,  # RTK_expr
    0.0,  # Aprot_plasma
    0.0,  # Aprot_tissue
    0.0,  # Artk_plasma
    0.0   # Artk_tissue
])



# ----------------------------
# Drug helper functions
# ----------------------------
def bolus_times(regimen):
    if regimen['n_pulses'] <= 0:
        return []
    if regimen['interval'] <= 0:
        return [regimen['start']]
    return [regimen['start'] + i * regimen['interval'] for i in range(regimen['n_pulses'])]

def conc_from_boluses(t, bolus_times_list, dose, half_life):
    if half_life <= 0:
        return dose if any(bt <= t + 1e-12 for bt in bolus_times_list) else 0.0
    k_el = np.log(2)/half_life
    return sum(dose * np.exp(-k_el*(t - bt)) for bt in bolus_times_list if t >= bt)

def emax_effect(C, Emax, EC50, hill=1.0):
    return Emax * (C ** hill) / (EC50 ** hill + C ** hill + 1e-12)

def hill(x, K, n=2):
    return x**n / (K**n + x**n) if K is not None and K > 0 else x

def two_compartment_drug(t, A_plasma, A_tissue, dose_events, k_elim, k12, k21):
    dose_input = sum(d for (ti, d) in dose_events if abs(t - ti) < 0.5)
    dA_plasma = -k_elim * A_plasma - k12 * A_plasma + k21 * A_tissue + dose_input
    dA_tissue = k12 * A_plasma - k21 * A_tissue
    return dA_plasma, dA_tissue

# ---------- compute drug exposure ----------
def compute_drug_exposure(sol_t, sol_y, idx_Aprot_tissue=13, idx_Artk_tissue=15):
    Cprot = sol_y[idx_Aprot_tissue, :]
    Crtk = sol_y[idx_Artk_tissue, :]
    return {
        'AUC_PROT': np.trapz(Cprot, sol_t),
        'AUC_RTK': np.trapz(Crtk, sol_t),
        'CMAX_PROT': np.max(Cprot),
        'CMAX_RTK': np.max(Crtk)
    }

def compute_harm_index(exposures, dose, t_sim, ref_cmax=10.0, w_auc=0.7, w_peak=0.3):
    ref_auc = dose * t_sim if (dose * t_sim) != 0 else 1.0
    auc = exposures['AUC_PROT'] + exposures['AUC_RTK']
    cmax = max(exposures['CMAX_PROT'], exposures['CMAX_RTK'])
    norm_auc = auc / ref_auc 
    norm_cmax = cmax / ref_cmax
    return w_auc * norm_auc + w_peak * norm_cmax

def compute_persister_index_from_solution(sol_t, sol_y, idx_N_pers=9, idx_N_bulk=8):
    N_pers = sol_y[idx_N_pers, :]
    N_bulk = sol_y[idx_N_bulk, :]
    f_pers = N_pers / (N_pers + N_bulk + 1e-12)
    PI_timeavg = np.trapz(f_pers, sol_t) / (sol_t[-1]-sol_t[0])
    PI_final = f_pers[-1]
    return {'PI_timeavg': PI_timeavg, 'PI_final': PI_final}

def combined_score(sol_t, sol_y, dose, t_sim, lambda_harm=1.0):
    pers = compute_persister_index_from_solution(sol_t, sol_y)
    exposures = compute_drug_exposure(sol_t, sol_y)
    H = compute_harm_index(exposures, dose, t_sim)
    score = pers['PI_timeavg'] + lambda_harm * H
    return {
        'score': score,
        'PI_timeavg': pers['PI_timeavg'],
        'PI_final': pers['PI_final'],
        'AUC_PROT': exposures['AUC_PROT'],
        'AUC_RTK': exposures['AUC_RTK'],
        'H': H
    }

# ----------------------------
# ODE system
# ----------------------------
# ----------------------------
# ODE system with adaptive signaling
# ----------------------------
def melanoma_ode(t, y, params, regimens):
    """
    ODE system for melanoma with PROTAC + RTK inhibition.
    Patient-specific parameters are read from `params`.
    """
    (RTK, RAS, RAF, MEK, MAPKpp, AKT_p, cFos, CyclinD,
     N_bulk, N_pers, DUSP, RTK_expr,
     Aprot_plasma, Aprot_tissue, Artk_plasma, Artk_tissue) = y

    N_tot = N_bulk + N_pers

    # ---- PK ----
    protac_doses = [(bt, regimens['protac']['dose']) for bt in bolus_times(regimens['protac'])]
    rtk_doses   = [(bt, regimens['rtk']['dose']) for bt in bolus_times(regimens['rtk'])]
    dAprot_plasma, dAprot_tissue = two_compartment_drug(t, Aprot_plasma, Aprot_tissue,
                                                        protac_doses, params['k_elim_protac'],
                                                        params['k12_protac'], params['k21_protac'])
    dArtk_plasma, dArtk_tissue = two_compartment_drug(t, Artk_plasma, Artk_tissue,
                                                      rtk_doses, params['k_elim_rtk'],
                                                      params['k12_rtk'], params['k21_rtk'])
    C_protac = Aprot_tissue
    C_rtk    = Artk_tissue

    # ---- PD ----
    E_protac = emax_effect(C_protac,
                           params.get('Emax_protac', 1.0),
                           params.get('EC50_protac', 1.0),
                           params.get('hill_protac', 1.0))
    E_rtk = emax_effect(C_rtk,
                        params.get('Emax_rtk', 1.0),
                        params.get('EC50_rtk', 1.0),
                        params.get('hill_rtk', 1.0))

    # ---- Adaptive / cluster scaling ----
    MAPK_baseline = params.get('MAPK_baseline', 0.3)
    AKT_baseline  = params.get('AKT_baseline', 0.4)
    adapt_strength = params.get('adapt_strength', 0.4)

    adapt_factor_rtk    = np.clip(1 + adapt_strength * (MAPK_baseline - MAPKpp), 0.5, 1.5)
    adapt_factor_raf    = np.clip(1 - 0.3 * adapt_strength * (MAPKpp - MAPK_baseline), 0.5, 1.5)
    adapt_factor_mek    = np.clip(1 - 0.2 * adapt_strength * E_rtk, 0.7, 1.2)
    adapt_factor_growth = np.clip(1 - 0.5 * E_protac, 0.3, 1.0)
    adapt_factor_pers   = np.clip(1 + 0.5 * (AKT_p - AKT_baseline), 0.5, 1.5)

    # ---- Base rates ----
    k_rtk_act  = params['k_rtk_act'] * adapt_factor_rtk
    k_raf_act  = params['k_raf_act'] * adapt_factor_raf
    k_mek_act  = params['k_mek_act'] * adapt_factor_mek
    r_bulk     = params['r_growth'] * adapt_factor_growth
    r_pers     = params.get('r_pers', 0.001 * params['r_growth']) * adapt_factor_pers

    k_rtk_deact = params['k_rtk_deact']
    k_ras_act   = params['k_ras_act']
    k_ras_deact = params['k_ras_deact']
    k_raf_deact = params['k_raf_deact']
    k_mek_deact = params['k_mek_deact']
    k_act       = params['k_act']
    k_deact     = params['k_deact']
    k_AKT_act   = params.get('k_AKT_act', 0.8)
    k_AKT_deact = params.get('k_AKT_deact', 0.3)
    k_cross     = params.get('k_cross', 0.5)
    k_rtk_syn   = params.get('k_rtk_syn', 0.02)
    k_rtk_deg   = params.get('k_rtk_deg', 0.01)
    k_dusp_prod = params['k_dusp_prod']
    k_dusp_decay= params['k_dusp_decay']
    k_dusp_inhib= params['k_dusp_inhib']
    k_cFos      = params['k_cFos']
    k_CyclinD   = params['k_CyclinD']
    k_decay     = params['k_decay']
    K           = params['K']

    # ---- Patient-specific rates ----
    k_conv0 = params.get('Pers_conv', params.get('k_conv0', 0.05))
    k_rev0  = params.get('Reversion', params.get('k_rev0', 0.02))
    RTK_expr0 = params.get('RTK_expr', RTK_expr)

    # ---- Signaling / transitions ----
    S_mapk = hill(MAPKpp, params.get('MAPK_50_up',0.5), n=2)
    dRTK_expr_dt = k_rtk_syn * (1 - S_mapk) - k_rtk_deg * RTK_expr
    dRTK_dt      = k_rtk_act * RTK_expr0 * (1 - RTK) - k_rtk_deact * RTK - params.get('k_inhib_rtk',1.5) * E_rtk * RTK
    dRAS_dt      = k_ras_act * RTK * (1 - E_rtk) * (1 - RAS) - k_ras_deact * RAS
    dRAF_dt      = k_raf_act * RAS * (1 - RAF) - k_raf_deact * RAF - C_protac * RAF
    dMEK_dt      = k_mek_act * RAF * (1 - MEK) - k_mek_deact * MEK
    dMAPKpp_dt   = k_act * MEK * (1 - MAPKpp) - k_deact * MAPKpp - k_dusp_inhib * DUSP * MAPKpp
    dAKT_dt      = (k_AKT_act * RTK + k_cross * (1 - MAPKpp)) * (1 - AKT_p) - k_AKT_deact * AKT_p
    dDUSP_dt     = k_dusp_prod * MAPKpp - k_dusp_decay * DUSP
    dcFos_dt     = k_cFos * MAPKpp - k_decay * cFos
    dCyclinD_dt  = k_CyclinD * cFos - k_decay * CyclinD

    # ---- Persister conversion ----
    conv_AKT_term = hill(AKT_p, params.get('AKT_50',0.5), params.get('nA',2))
    conv_MAPK_term= params.get('MAPK_50',0.3)**params.get('nM',2) / (params.get('MAPK_50',0.3)**params.get('nM',2) + MAPKpp**params.get('nM',2))
    k_conv = k_conv0 * conv_AKT_term * conv_MAPK_term
    k_rev = (
        k_rev0
        * hill(MAPKpp, params.get('MAPK_50', 0.3), n=params.get('nM', 2))
        * (1 + 0.5 * E_protac)
    )

    # ---- Population dynamics ----
    dN_bulk = r_bulk*N_bulk*(1 - N_tot/K) - E_protac*N_bulk - k_conv*N_bulk + k_rev*N_pers
    dN_pers = r_pers*N_pers*(1 - N_tot/K) - (0.2*E_protac + 0.2*E_rtk + 0.2*E_protac*E_rtk)*N_pers + k_conv*N_bulk - k_rev*N_pers

    dydt = [
        dRTK_dt, dRAS_dt, dRAF_dt, dMEK_dt, dMAPKpp_dt, dAKT_dt,
        dcFos_dt, dCyclinD_dt, dN_bulk, dN_pers, dDUSP_dt, dRTK_expr_dt,
        dAprot_plasma, dAprot_tissue, dArtk_plasma, dArtk_tissue
    ]
    return dydt


# ----------------------------
# Initial conditions
# ----------------------------
y0 = [0.1,0,0,0,0.2,0.1,0,0,0.9,0.1,0,1.0,0,0,0,0]

# ----------------------------
# Drug regimens
# ----------------------------
regimens = {
    'protac': {'dose':1.0, 'start':0, 'interval':24, 'n_pulses':2, 'half_life':6.0},
    'rtk': {'dose':1.0, 'start':0, 'interval':24, 'n_pulses':2, 'half_life':8.0}
}

# ----------------------------
# Solve ODE
# ----------------------------
t_span = (0, 120)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# ----------------------------
# State labels
# ----------------------------
states = ['RTK','RAS','RAF','MEK','MAPKpp','AKT_p',
          'cFos','CyclinD','N_bulk','N_pers','DUSP','RTK_expr',
          'Aprot_plasma','Aprot_tissue','Artk_plasma','Artk_tissue']


# ----------------------------
# Guard
# ----------------------------
if __name__ == "__main__":
    print("melanoma_model.py defines the ODE system - run driver.py instead.")

# ----------------------------
# Simulation wrapper
# ----------------------------
def simulate(regimen, patient_params, params=params, y0=None, t_span=(0,120), t_eval=None):
    """
    Run melanoma model for a single patient and single regimen.
    
    Arguments:
        regimen: dict with 'protac' and 'rtk' dosing info
        patient_params: dict of patient-specific parameters
        params: model parameters (default from melanoma_model)
        y0: initial conditions (default_y0)
        t_span: tuple, simulation time span
        t_eval: array of times to evaluate
    
    Returns:
        dict with PI_timeavg, PI_final, AUCs, harm_index, COS, and optional time_series DataFrame
    """
    import numpy as np
    from scipy.integrate import solve_ivp
    import pandas as pd

    # Defaults
    if y0 is None:
        y0 = default_y0.copy()
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 500)

    # Adjust initial conditions from patient_params
    y0_sim = y0.copy()
    if "MAPKpp" in patient_params:
        y0_sim[4] = patient_params["MAPKpp"]
    if "AKT" in patient_params:
        y0_sim[5] = patient_params["AKT"]
    if "RTK_expr" in patient_params:
        y0_sim[11] = patient_params["RTK_expr"]
    if "Init_pers_frac" in patient_params:
        N_tot = y0_sim[8] + y0_sim[9]
        y0_sim[9] = N_tot * patient_params["Init_pers_frac"]
        y0_sim[8] = N_tot - y0_sim[9]

    # Run ODE solver
    sol = solve_ivp(
        lambda t, y: melanoma_ode(t, y, params, regimen),
        t_span,
        y0_sim,
        t_eval=t_eval,
        method='RK45'
    )

    # Compute metrics
    pers = compute_persister_index_from_solution(sol.t, sol.y)
    exposures = compute_drug_exposure(sol.t, sol.y)
    H = compute_harm_index(exposures, dose=regimen['protac']['dose'], t_sim=t_span[1])
    
    # Composite Optimization Score (COS)
    w_PI_timeavg = 0.5
    w_PI_final = 0.3
    w_harm = 0.15
    w_exposure = 0.05

    total_auc = max(exposures['AUC_PROT'] + exposures['AUC_RTK'], 1e-6)
    harm_norm = H / total_auc
    exposure_norm = total_auc / 50.0

    COS = (
        (1 - pers['PI_timeavg']) * w_PI_timeavg +
        (1 - pers['PI_final']) * w_PI_final -
        (harm_norm * w_harm + exposure_norm * w_exposure)
    )

    # Optional time series for saving
    ts_df = pd.DataFrame(sol.y.T, columns=states)
    ts_df.insert(0, 't', sol.t)

    return {
        "regimen": regimen["name"],
        "PI_timeavg": pers['PI_timeavg'],
        "PI_final": pers['PI_final'],
        "AUC_PROT": exposures['AUC_PROT'],
        "AUC_RTK": exposures['AUC_RTK'],
        "harm_index": H,
        "COS": COS,
        "time_series": ts_df
    }


# -------------------------------
# Summarize across patient clusters
# -------------------------------
def summarize(patient_clusters, regimens, N_patients=10):
    """
    Summarize model outputs across clusters, patients, and regimens.

    Arguments:
        patient_clusters: dict of clusters with parameter ranges
        regimens: list of regimen dicts
        N_patients: number of virtual patients per cluster

    Returns:
        DataFrame with mean metrics per cluster x regimen
    """
    import pandas as pd

    all_results = []

    for cluster_name, cluster_info in patient_clusters.items():
        print(f"Simulating cluster: {cluster_name}")

        for patient_id in range(1, N_patients + 1):
            patient_params = sample_patient(cluster_info["params"])

            for reg in regimens:
                # run simulate for this patient + regimen
                result = simulate(reg, patient_params)

                # add patient and cluster info
                result["cluster"] = cluster_name
                result["patient_id"] = patient_id

                # also store patient parameters if desired
                result.update(patient_params)

                all_results.append(result)

    # convert to DataFrame
    df_all = pd.DataFrame(all_results)

    # compute summary metrics per cluster x regimen
    summary = (
        df_all.groupby(["cluster", "regimen"])
        .agg({
            "PI_timeavg": "mean",
            "PI_final": "mean",
            "AUC_PROT": "mean",
            "AUC_RTK": "mean",
            "COS": "mean"
        })
        .reset_index()
    )

    return summary
