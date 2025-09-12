# ── stdlib ────────────────────────────────────────────────────────────
import argparse, datetime, os, sys
from pathlib import Path
from cProfile import Profile
from pstats  import Stats, SortKey
from typing  import List

# ── third‑party ───────────────────────────────────────────────────────
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# ── internal packages ────────────────────────────────────────────────
sys.path.append(os.path.abspath(".."))  # adjust if needed


from src.lm_mono_debias import lm_fix_alpha, lm_change_alpha_every_iter, lm_mcar_extended, lm_mono_debias_budget_constrained_obtain_alpha_mcar_cov00, lm_mono_debias_estimate_mcar_crossfit, lm_mono_debias_budget_constrained_obtain_alpha_mar_cov00, lm_mono_debias_estimate_mar_crossfit  # Algorithm‑1 wrapper
from src.utils import (
    get_device, set_global_seed, sample_split, wald_ci         # <-- our helpers
)
from src.models import AlphaModel
from src.data_generation import lm_generate_obs_data_mcar, lm_generate_obs_data_mar, lm_generate_complete_data, general_generate_mcar, general_generate_mar
from src.estimators import (
    lm_fit_ols,                                    # OLS regression
    lm_fit_wls,                                    # WLS regression
    lm_build_all_psi,                             # compute ψ-values (no grads)
    lm_build_all_psi_weighted,                     # compute ψ-values (with weights)
    general_build_all_phi,                        # evaluate φ on a fixed alpha
    general_build_all_phi_mar,                    # evaluate φ on a fixed alpha (MAR)
    general_build_all_phi_function,               # return callables φ1/φ2/φ3
    general_build_all_phi_function_mar,           # return callables φ1/φ2/φ3 (MAR)
    general_estimate_moments_mcar,                # E[ψψᵀ], E[φ_jφ_jᵀ], Cov(ψ,φ_j)
    general_estimate_moments_mar,                 # E[ψψᵀ], E[φ_jφ_jᵀ], Cov(ψ,φ_j) (MAR)
    general_estimate_moments_function_mcar,       # return moment_fn(alpha_vec)
    general_estimate_moments_function_mar,        # return moment_fn(alpha_vec) (MAR)
    general_estimate_m_matrix_mcar,               # estimate M^(1)
    general_estimate_m_matrix_mar,                # estimate M^(1) (MAR)
    general_estimate_variance_mcar,               # estimate Cov(θ̂)
    general_estimate_variance_mar,                # estimate Cov(θ̂) (MAR)
    general_get_trace_variance_function_alpha_mcar,# return g(alpha1)
    general_get_cov00_function_alpha_mcar,       # return g00(alpha1)
    general_get_trace_variance_function_alpha_mar,# return g(alpha1) (MAR)
    general_get_cov00_function_alpha_mar,        # return g00(alpha1) (MAR)
    search_alpha_mcar,                            # public API: "golden" | "adam"
    search_alpha_mcar_trace,                            # public API: "golden" | "adam"
    train_alpha_with_lagrangian,                # train a neural network to predict α₁
    train_alpha_aug_lagrange,                   # train a neural network to predict α₁
    train_alpha_with_penalty,                # train a neural network to predict α₁
    train_alpha_aug_lagrange_trace,          # train a neural network to predict α₁
    lm_mono_debias_estimate,                 # 3-fold cross-fit efficient θ̂
)

# Paths and random seed
ROOT = Path.cwd()                 # project root (where the notebook lives)
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "US_wine_extended_full.csv"
SEED = 42

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Load & preprocess wine.data
# -----------------------------
n = 10000  # maximum number of rows to use

# Load CSV and drop rows with missing values in the relevant columns
use_cols = ["gpt_point", "points", "price", "province", "predicted_score"]
data = pd.read_csv(CSV_PATH, usecols=use_cols)
data = data.dropna(subset=use_cols)

# Take the first n rows and shuffle (equivalent to MATLAB randperm)
data = data.head(n).sample(frac=1, random_state=SEED).reset_index(drop=True)
n = len(data)  # update n to reflect actual number of rows after filtering
data['is_ca'] = (data['province'] == 'California').astype('int64')
data['is_wa'] = (data['province'] == 'Washington').astype('int64')
data['is_or'] = (data['province'] == 'Oregon').astype('int64')
data['is_ny'] = (data['province'] == 'New York').astype('int64')
# -----------------------------
# Convert columns to tensors
# -----------------------------
# X, W1, W2 will be column vectors of shape (n,1)
# Y, V will also be reshaped as (n,1) for consistency

W1 = torch.tensor(data["gpt_point"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
W1[W1 < 80] = 80.0  # enforce lower bound

W2 = torch.tensor(data["predicted_score"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
W2[W2 < 80] = 80.0
# Alternative: W2 = torch.full_like(W1, 90.0)

# Construct W2 = W1 + Normal noise
noise1 = torch.randn_like(W1) * 8.0
noise2 = torch.randn_like(W1) * 1.0

Y = torch.tensor(data["points"].to_numpy(), dtype=torch.float32).reshape(-1, 1)
X = torch.tensor(np.column_stack((np.log(data['price']), data['is_ca'], data['is_wa'], data['is_or'], data['is_ny'], np.ones(n))).astype(float), dtype=torch.float32)
#W2 = Y.clone()  # start with true Y
# W2 = W2 + noise2  # add noise
# W2[W2 < 80] = 80.0  # enforce lower bound
# V: indicator if W2 is closer to Y than W1 (absolute error comparison)
V = (torch.abs(W1 - Y) <= torch.abs(W2 - Y)).float()

# -----------------------------
# Parameter settings
# -----------------------------
params = {
    "n1": int(n * 0.1),  # Stage 1 sample size
    "n2": int(n * 0.9),  # Stage 2 sample size
    "alpha_init": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
}
alpha_level = 0.1
c_list = [5, 10]   # alternative: [5, 10, 20]
itr_num = 50

# -----------------------------
# OLS to get theta_star (no intercept)
# θ* = (∑ x_i y_i) / (∑ x_i^2)
# -----------------------------
denom = X.pow(2).sum().item()
if denom == 0:
    raise ValueError("X is all zeros, cannot perform no-intercept OLS.")



# -----------------------------
# Save packed results
# -----------------------------
wine_tensors = {"X": X, "Y": Y, "W1": W1, "W2": W2, "V": V}
save_path = DATA_DIR / "wine_data.pt"
torch.save(wine_tensors, save_path)

print("Saved:", str(save_path))
print("Keys:", list(wine_tensors.keys()))
print("n =", n, "| shapes:",
      "X", tuple(X.shape), "Y", tuple(Y.shape),
      "W1", tuple(W1.shape), "W2", tuple(W2.shape), "V", tuple(V.shape))


# pick device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load
batch = torch.load(save_path, map_location="cpu")

# move tensors to GPU (or keep on CPU if no cuda)
X  = batch["X"].to(device)
Y  = batch["Y"].to(device)
W1 = batch["W1"].to(device)
W2 = batch["W2"].to(device)
V  = batch["V"].to(device)
theta_star = lm_fit_ols(X, Y)
print("theta_star (no-intercept OLS):", theta_star)
W2 = X @ theta_star.view(-1, 1) + torch.randn_like(W2) * 2
W1 = X @ theta_star.view(-1, 1) + torch.randn_like(W1) * 2
print(X.device, Y.device, W1.device, W2.device, V.device)






from typing import Dict
import torch
from tqdm import tqdm

def lm_fix_alpha(
    *,
    # --------- prepared dataset ----------
    X: torch.Tensor,
    Y: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    V: torch.Tensor,
    # --------- batch sizes ---------------
    n1: int,             # Stage-1 size
    n2_per_rep: int,     # Stage-2 size sampled each rep
    reps: int,
    # --------- reference / CI ------------
    theta_star: torch.Tensor | float,
    alpha_level: float,
    tau: float,
    c: float,
    alpha_init: torch.Tensor,           # (3,)
    # --------- misc ----------------------
    seed: int = 42,
) -> Dict[str, float | torch.Tensor]:

    device, dtype = X.device, X.dtype
    torch.manual_seed(seed)

    N = X.shape[0]
    n2_pool_size = N - n1

    if N < n1 + n2_per_rep:
        raise ValueError(f"Dataset too small: need >= n1+n2_per_rep={n1+n2_per_rep}, got {N}")
    if n2_pool_size < n2_per_rep:
        raise ValueError(f"Stage-2 pool too small: need >= {n2_per_rep}, got {n2_pool_size}")

    # -------- Stage-1: obtain α* on a fixed split ----------
    idx_perm  = torch.randperm(N, device=device)
    idx1      = idx_perm[:n1]
    idx2_pool = idx_perm[n1:]  # fixed Stage-2 pool

    X1, Y1, W1_1, W2_1, V1 = X[idx1], Y[idx1], W1[idx1], W2[idx1], V[idx1]

    # R1 ~ Categorical(alpha_init)
    alpha_init = alpha_init.to(device=device, dtype=dtype)
    R1 = torch.multinomial(alpha_init.view(1, -1).expand(n1, -1), 1) + 1

    Y1_masked = Y1.clone().float()
    V1_masked = V1.clone().float()
    r1 = R1.view(-1, 1)
    Y1_masked[r1 == 2] = torch.nan
    Y1_masked[r1 == 3] = torch.nan
    V1_masked[r1 == 3] = torch.nan

    # learn α* (MCAR) and α_model (MAR)
    alpha_opt, cov00_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mcar_cov00(
        X1, Y1_masked, W1_1, W2_1, V1_masked, R1, tau=tau, c=c, method="mlp"
    )
    alpha_model_opt, cov00_model_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mar_cov00(
        X1, Y1_masked, W1_1, W2_1, V1_masked, R1, tau=tau, c=c, method="mlp"
    )

    # baseline α (α2=0)
    alpha1_base = float(tau) / float(c)
    alpha_baseline = torch.tensor(
        [alpha1_base, 0.0, 1.0 + (c - 1.0) * alpha1_base - tau],
        device=device, dtype=dtype
    ).clamp(min=0)
    alpha_baseline = alpha_baseline / alpha_baseline.sum().clamp_min(1e-12)

    # -------- containers ----------
    err_opt, err_mar, err_base, err_ols = [], [], [], []
    len_opt, len_mar, len_base, len_ols = [], [], [], []
    cov_opt, cov_mar, cov_base, cov_ols = [], [], [], []

    theta_star_val = float(theta_star[0].item() if torch.is_tensor(theta_star) else theta_star)

    # =================== Stage-2 loop ==================
    for rep in tqdm(range(reps), total=reps):
        torch.manual_seed(seed + rep)

        # sample n2_per_rep from the fixed Stage-2 pool
        subperm = torch.randperm(n2_pool_size, device=device)
        idx2 = idx2_pool[subperm[:n2_per_rep]]

        X2, Y2, W1_2, W2_2, V2 = X[idx2], Y[idx2], W1[idx2], W2[idx2], V[idx2]

        # ---- Batch-A: MCAR with alpha_opt ----
        alphaA = alpha_opt.to(device).clamp(min=0)
        alphaA = alphaA / alphaA.sum().clamp_min(1e-12)
        R2A = torch.multinomial(alphaA.view(1, -1).expand(n2_per_rep, -1), 1) + 1

        Y2A_obs = Y2.clone().float()
        V2A_obs = V2.clone().float()
        rA = R2A.view(-1, 1)
        Y2A_obs[rA == 2] = torch.nan
        Y2A_obs[rA == 3] = torch.nan
        V2A_obs[rA == 3] = torch.nan

        theta_opt_vec, cov_opt_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2, Y2A_obs, W1_2, W2_2, V2A_obs, R2A, alpha=alpha_opt, method="mlp"
        )
        se_opt_1 = torch.sqrt(cov_opt_mat[0, 0] / n2_per_rep).item()
        ci_opt_low, ci_opt_high = wald_ci(theta_opt_vec[0].item(), se_opt_1, alpha_level)

        # ---- Batch-MAR: MAR with alpha_model_opt ----
        with torch.no_grad():
            alpha_all = alpha_model_opt(X2, W1_2, W2_2).to(device)
            if alpha_all.ndim != 2 or alpha_all.shape[1] != 3:
                raise ValueError("alpha_model_opt must return (n,3)")
            alpha_all = alpha_all.clamp(min=0)
            alpha_all = alpha_all / alpha_all.sum(dim=1, keepdim=True).clamp_min(1e-12)
        R2M = torch.multinomial(alpha_all, 1) + 1

        Y2M_obs = Y2.clone().float()
        V2M_obs = V2.clone().float()
        rM = R2M.view(-1, 1)
        Y2M_obs[rM == 2] = torch.nan
        Y2M_obs[rM == 3] = torch.nan
        V2M_obs[rM == 3] = torch.nan

        theta_mar_vec, cov_mar_mat = lm_mono_debias_estimate_mar_crossfit(
            X2, Y2M_obs, W1_2, W2_2, V2M_obs, R2M, alpha_fn=alpha_model_opt, method="mlp"
        )
        se_mar_1 = torch.sqrt(cov_mar_mat[0, 0] / n2_per_rep).item()
        ci_mar_low, ci_mar_high = wald_ci(theta_mar_vec[0].item(), se_mar_1, alpha_level)

        # ---- Batch-B: MCAR baseline ----
        R2B = torch.multinomial(alpha_baseline.view(1, -1).expand(n2_per_rep, -1), 1) + 1

        Y2B_obs = Y2.clone().float()
        V2B_obs = V2.clone().float()
        rB = R2B.view(-1, 1)
        Y2B_obs[rB == 2] = torch.nan
        Y2B_obs[rB == 3] = torch.nan
        V2B_obs[rB == 3] = torch.nan

        theta_base_vec, cov_base_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2, Y2B_obs, W1_2, W2_2, V2B_obs, R2B, alpha=alpha_baseline, method="mlp"
        )
        se_base_1 = torch.sqrt(cov_base_mat[0, 0] / n2_per_rep).item()
        ci_base_low, ci_base_high = wald_ci(theta_base_vec[0].item(), se_base_1, alpha_level)

        # ---- OLS on observed Y ----
        mask_ols = ~torch.isnan(Y2B_obs).view(-1)
        X_ols = X2[mask_ols]
        Y_ols = Y2B_obs[mask_ols]
        n_ols = X_ols.shape[0]
        d_ols = X_ols.shape[1]
        theta_ols_vec = lm_fit_ols(X_ols, Y_ols)

        residuals = Y_ols - (X_ols @ theta_ols_vec.view(-1, 1))
        s_squared = (residuals.T @ residuals).item() / max(1, (n_ols - d_ols))
        inv_xtx = torch.inverse(X_ols.T @ X_ols)
        se_ols_1 = torch.sqrt(s_squared * inv_xtx[0, 0]).item()
        ci_ols_low, ci_ols_high = wald_ci(theta_ols_vec[0].item(), se_ols_1, alpha_level)

        # ---- accumulate ----
        err_opt .append(abs(theta_opt_vec[0].item()  - theta_star_val))
        err_mar .append(abs(theta_mar_vec[0].item()  - theta_star_val))
        err_base.append(abs(theta_base_vec[0].item() - theta_star_val))
        err_ols .append(abs(theta_ols_vec[0].item()  - theta_star_val))

        len_opt .append(ci_opt_high  - ci_opt_low)
        len_mar .append(ci_mar_high  - ci_mar_low)
        len_base.append(ci_base_high - ci_base_low)
        len_ols .append(ci_ols_high  - ci_ols_low)

        cov_opt .append(int(ci_opt_low  <= theta_star_val <= ci_opt_high))
        cov_mar .append(int(ci_mar_low  <= theta_star_val <= ci_mar_high))
        cov_base.append(int(ci_base_low <= theta_star_val <= ci_base_high))
        cov_ols .append(int(ci_ols_low  <= theta_star_val <= ci_ols_high))

    # -------- return --------
    return dict(
        alpha_opt     = ((alpha_opt.detach().cpu() * 1e4).round() / 1e4).tolist(),
        cov00_opt     = float(cov00_opt),
        mean_l2_opt   = float(sum(err_opt)  / reps),
        mean_l2_mar   = float(sum(err_mar)  / reps),
        mean_l2_base  = float(sum(err_base) / reps),
        mean_l2_ols   = float(sum(err_ols)  / reps),
        mean_len_opt  = float(sum(len_opt)  / reps),
        mean_len_mar  = float(sum(len_mar)  / reps),
        mean_len_base = float(sum(len_base) / reps),
        mean_len_ols  = float(sum(len_ols)  / reps),
        covg_opt      = float(sum(cov_opt)  / reps),
        covg_mar      = float(sum(cov_mar)  / reps),
        covg_base     = float(sum(cov_base) / reps),
        covg_ols      = float(sum(cov_ols)  / reps),
    )


out = lm_fix_alpha(
    X=X, Y=Y, W1=W1, W2=W2, V=V,
    n1=2000,            # Stage-1 样本数
    n2_per_rep=7000,    # 每次 Stage-2 随机抽取 8000
    reps=10,            # 重复次数
    theta_star=theta_star,
    alpha_level=0.1,
    tau=3.0,
    c=5.0,
    alpha_init=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device),
    seed=42
)

for k, v in out.items():
    print(f"{k:15s} : {v}")