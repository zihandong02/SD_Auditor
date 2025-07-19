import os, sys, numpy as np, pandas as pd
import datetime, json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.weightstats import _zconfint_generic, _zstat_generic

sys.path.append(os.path.abspath(".."))

# internal modules (from your src/ package)
from src.utils           import SEED, set_global_seed, dump_run_simple
from src.data_generation import lm_generate_obs_data_mcar
from src.mono_debais     import (
    lm_mono_debias_estimate_mcar_crossfit,
    lm_mono_debias_budget_constrained_obtain_alpha_mcar,
    lm_mono_debias_budget_constrained_obtain_alpha_mcar_var1
)
from src.baselines       import lm_ols_estimate
# ------------------------------------------------------------
# argument parser
# ------------------------------------------------------------

# ------------------- 1. Global settings -------------------
SEED = 42
set_global_seed(SEED)           # Your own helper that sets both NumPy & Python RNG





def lm_mcar(
    *,
    # ----------------  batch sizes ----------------
    n1: int,
    n2: int,
    reps: int,
    # ----------------  model dims -----------------
    d_x: int,
    d_u1: int,
    d_u2: int,
    # ----------------  model parameters -----------
    theta_star: np.ndarray,
    beta1_star: np.ndarray,
    beta2_star: np.ndarray,
    # ----------------  noise ----------------------
    sigma_eps: float,
    # ----------------  MCAR / CI ------------------
    alpha_level: float,
    tau: float,
    c: float,
    alpha_init: np.ndarray,
    # ----------------  misc -----------------------
    seed: int = 42
) -> dict:
    """
    Full Stage-1 + Stage-2 MCAR pipeline.

    Returns
    -------
    dict with
        alpha_opt, trace_opt
        mean_l2_opt / base / ols
        mean_len_opt / base
        covg_opt / base
    """
    # -------------- Stage-1 : choose α* --------------------
    set_global_seed(seed)

    X1, Y1, W1_1, W2_1, V1, R1 = lm_generate_obs_data_mcar(
        n=n1,
        d_x=d_x, d_u1=d_u1, d_u2=d_u2,
        theta_star=theta_star,
        beta1_star=beta1_star,
        beta2_star=beta2_star,
        alpha=alpha_init,
        Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
        sigma_eps=sigma_eps,
    )

    alpha_opt, trace_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mcar_var1(
        X1, Y1, W1_1, W2_1, V1, R1,
        tau=tau, c=c, method="mlp"
    )

    # baseline α with α₂ = 0
    alpha1_base   = tau / c
    alpha_baseline = np.array([alpha1_base,
                               0.0,
                               1.0 + (c - 1.0) * alpha1_base - tau])

    # -------------- Stage-2 metrics containers --------------
    err_opt, err_base, err_ols = [], [], []
    len_opt, len_base          = [], []
    cov_opt, cov_base          = [], []

    for rep in range(reps):
        set_global_seed(seed + rep)

        # ----- Batch-A : α_opt ---------------------------------------
        X2A, Y2A, W1A, W2A, V2A, R2A = lm_generate_obs_data_mcar(
            n=n2,
            d_x=d_x, d_u1=d_u1, d_u2=d_u2,
            theta_star=theta_star,
            beta1_star=beta1_star,
            beta2_star=beta2_star,
            alpha=alpha_opt,
            Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
            sigma_eps=sigma_eps,
        )
        theta_opt, cov_opt_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2A, Y2A, W1A, W2A, V2A, R2A,
            alpha=alpha_opt, method="mlp"
        )
        theta_opt  = np.asarray(theta_opt)
        cov_opt_mat = np.asarray(cov_opt_mat)
        se_opt_1 = np.sqrt(cov_opt_mat[0, 0] / n2)
        ci_opt_low, ci_opt_high = _zconfint_generic(
            theta_opt[0], se_opt_1,
            alpha=alpha_level, alternative="two-sided"
        )

        # ----- Batch-B : α_baseline ------------------------------
        X2B, Y2B, W1B, W2B, V2B, R2B = lm_generate_obs_data_mcar(
            n=n2,
            d_x=d_x, d_u1=d_u1, d_u2=d_u2,
            theta_star=theta_star,
            beta1_star=beta1_star,
            beta2_star=beta2_star,
            alpha=alpha_baseline,
            Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
            sigma_eps=sigma_eps,
        )
        theta_base, cov_base_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2B, Y2B, W1B, W2B, V2B, R2B,
            alpha=alpha_baseline, method="mlp"
        )
        theta_base  = np.asarray(theta_base)
        cov_base_mat = np.asarray(cov_base_mat)
        se_base_1 = np.sqrt(cov_base_mat[0, 0] / n2)
        ci_base_low, ci_base_high = _zconfint_generic(
            theta_base[0], se_base_1,
            alpha=alpha_level, alternative="two-sided"
        )

        # ----- OLS reference ------------------------------------
        theta_ols = lm_ols_estimate(X2B, Y2B)

        # ----- accumulate metrics ------------------------------
        err_opt .append(np.linalg.norm(theta_opt  - theta_star))
        err_base.append(np.linalg.norm(theta_base - theta_star))
        err_ols .append(np.linalg.norm(theta_ols  - theta_star))

        len_opt .append(ci_opt_high  - ci_opt_low)
        len_base.append(ci_base_high - ci_base_low)

        cov_opt .append(int(ci_opt_low  <= theta_star[0] <= ci_opt_high))
        cov_base.append(int(ci_base_low <= theta_star[0] <= ci_base_high))

    # ---------------- aggregate & return ---------------------
    return dict(
        alpha_opt=np.round(alpha_opt, 4),
        trace_opt=trace_opt,
        mean_l2_opt=np.mean(err_opt),
        mean_l2_base=np.mean(err_base),
        mean_l2_ols=np.mean(err_ols),
        mean_len_opt=np.mean(len_opt),
        mean_len_base=np.mean(len_base),
        covg_opt=np.mean(cov_opt),
        covg_base=np.mean(cov_base),
    )





# 1) Define the grid of τ values you want to explore
tau_values = [2.0, 3.0, 4.0]


d_x = 5
d_u1 = 5
d_u2 = 5
common = dict(
    n1=2000, n2=10000, reps=10,
    d_x=d_x, d_u1=d_u1, d_u2=d_u2,
    theta_star = np.arange(1, d_x  + 1) * 0.2,   # length d_x
    beta1_star = np.arange(1, d_u1 + 1) * 1.0,   # length d_u1
    beta2_star = np.arange(1, d_u2 + 1) * -0.4,  # length d_u2
    sigma_eps = 1.0,
    alpha_level = 0.1,
    c = 6.0,
    alpha_init = np.array([1/3, 1/3, 1/3])
)

# ------------------------------------------------------------------
# 1)  run lm_mcar for each τ  →  build DataFrame `df`
# ------------------------------------------------------------------
rows = []
for τ in tau_values:
    res = lm_mcar(tau=τ, **common)          # type: ignore[arg-type]
    res["tau"] = τ
    rows.append(res)

df = pd.DataFrame(rows).set_index("tau").round(4)

# ------------------------------------------------------------------
# 2)  save summary + parameters, capture returned folder
# ------------------------------------------------------------------
params   = {"tau_values": tau_values, "common_args": common}
time_dir = dump_run_simple(df=df, params=params)     # ← returns out-dir path

# ------------------------------------------------------------------
# 3)  Figure: CI length vs τ   (stored inside the same folder)
# ------------------------------------------------------------------
plt.figure(figsize=(7, 3.5))
plt.plot(df.index, df["mean_len_opt"],  marker="o", label="opt-alpha")
plt.plot(df.index, df["mean_len_base"], marker="o", label="base-alpha")
plt.xlabel(r"$\tau$")
plt.ylabel("CI length (first coord)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(time_dir, "ci_length_vs_tau.pdf"))
plt.close()

# ------------------------------------------------------------------
# 4)  Figure: Coverage vs τ    (stored inside the same folder)
# ------------------------------------------------------------------
plt.figure(figsize=(7, 3.5))
plt.plot(df.index, df["covg_opt"],  marker="o", label="opt-alpha")
plt.plot(df.index, df["covg_base"], marker="o", label="base-alpha")
plt.axhline(1 - common["alpha_level"], ls="--", color="gray")   # nominal level
plt.ylim(0.5, 1.05)
plt.xlabel(r"$\tau$")
plt.ylabel("Coverage (first coord)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(time_dir, "coverage_vs_tau.pdf"))
plt.close()