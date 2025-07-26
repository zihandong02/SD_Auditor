"""
mono_bias.py
============
High-level MCAR mono-debias pipeline

Public API
----------
1) lm_ols_estimate                                – OLS baseline (Torch)
2) lm_mono_debias_estimate_mcar_crossfit          – 3-fold efficient θ̂ (given alpha)
3) lm_mono_debias_budget_constrained_obtain_alpha_mcar
                                                 – 2-fold search for alpha* (Alg-4)
4) lm_mcar                                        – Full Alg-1 (Stage-1 + Stage-2)
"""

from __future__ import annotations
from typing import Sequence, Literal, Dict, List, Tuple

import torch
from torch import Tensor
from torch.distributions import Normal
from tqdm import tqdm

from .utils import (
    get_device, set_global_seed, sample_split, wald_ci         # <-- our helpers
)
from .data_generation import lm_generate_obs_data_mcar
from .estimators import (
    lm_fit_ols,                                    # OLS regression
    lm_fit_wls,                                    # WLS regression
    lm_build_all_psi,                             # compute ψ-values (no grads)
    general_build_all_phi,                        # evaluate φ on a fixed alpha
    general_build_all_phi_function,               # return callables φ1/φ2/φ3
    general_estimate_moments_mcar,                # E[ψψᵀ], E[φ_jφ_jᵀ], Cov(ψ,φ_j)
    general_estimate_moments_function_mcar,       # return moment_fn(alpha_vec)
    general_estimate_m_matrix_mcar,               # estimate M^(1)
    general_estimate_variance_mcar,               # estimate Cov(θ̂)
    general_get_trace_variance_function_alpha_mcar,# return g(alpha1)
    general_get_cov00_function_alpha_mcar,       # return g00(alpha1)
    search_alpha_mcar,                            # public API: "golden" | "adam"
    lm_mono_debias_estimate_mcar,                 # 3-fold cross-fit efficient θ̂
)


# ======================================================================
# 1) 3-fold cross-fit efficient estimator  (Algorithm 3)
# ======================================================================
@torch.no_grad()  # cross-fit is purely empirical – gradients not needed
def lm_mono_debias_estimate_mcar_crossfit(
    X: Tensor, Y: Tensor,
    W1: Tensor, W2: Tensor, V: Tensor,
    R: Tensor,                      # (n,) int64 values {1,2,3}
    alpha: Tensor,
    method: str = "mlp"
) -> Tuple[Tensor, Tensor]:
    """
    End-to-end 3-fold cross-fit pipeline (Steps 1–6) – Torch version.

    Returns
    -------
    theta_final : (d,)  Tensor   — cross-fitted efficient estimator
    cov_final   : (d,d) Tensor   — cross-fitted covariance estimate
    """

    # ---------- Step 1 : deterministic 3-way split ----------
    device, dtype = X.device, X.dtype
    n, d = X.shape
    idx1, idx2, idx3 = sample_split(n, 3, device=device)        # assumes Torch-compatible
    folds = [
        (idx1, idx2, idx3),   # rotation 1  (D1, D2, D3)
        (idx2, idx3, idx1),   # rotation 2
        (idx3, idx1, idx2)    # rotation 3
    ]

    # Constant-alpha functions (MCAR)
    alpha1_fn = lambda Xv, w1v, w2v: torch.full((len(w1v),), alpha[0],
                                               device=device, dtype=dtype)
    alpha2_fn = lambda Xv, w1v, w2v: torch.full((len(w1v),), alpha[1],
                                               device=device, dtype=dtype)
    alpha3_fn = lambda Xv, w1v, w2v: torch.full((len(w1v),), alpha[2],
                                               device=device, dtype=dtype)

    def subset(idxs: Tensor):
        """Return tensors restricted to `idxs` (1-D LongTensor)."""
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]

    theta_list: List[Tensor] = []
    cov_list:   List[Tensor] = []
    size_list:  List[int]    = []

    # ---------- 3-fold rotation loop ----------
    for D1, D2, D3 in folds:
        # Convert index arrays to Torch tensors for slicing
        D1_t = torch.as_tensor(D1, device=device, dtype=torch.long)
        D2_t = torch.as_tensor(D2, device=device, dtype=torch.long)
        D3_t = torch.as_tensor(D3, device=device, dtype=torch.long)

        # Fetch data subsets
        X1, Y1, W11, W21, V1, R1 = subset(D1_t)
        X2, Y2, W12, W22, V2, R2 = subset(D2_t)
        X3, Y3, W13, W23, V3, R3 = subset(D3_t)

        # ----- Step 2 : build ψ and φ on D3 ∩ {R = 1} -----
        mask_r1_D3 = (R3 == 1).squeeze(1)
        theta_pre = lm_fit_wls(
            X3[mask_r1_D3],
            Y3[mask_r1_D3])

        psi_1, psi_2, psi_3 = lm_build_all_psi(
            X3[mask_r1_D3], Y3[mask_r1_D3],
            W13[mask_r1_D3], W23[mask_r1_D3], V3[mask_r1_D3],
            theta_pre, method=method)

        phi_1, phi_2, phi_3 = general_build_all_phi(
            psi_2, psi_3,
            alpha)

        # ----- Step 3 : estimate covariance matrices on D2 ∩ {R = 1} -----
        mask_r1_D2 = (R2 == 1).squeeze(1)
        moments = general_estimate_moments_mcar(
            psi_1, phi_1, phi_2, phi_3,
            X2[mask_r1_D2], Y2[mask_r1_D2],
            W12[mask_r1_D2], W22[mask_r1_D2], V2[mask_r1_D2])

        # Estimate M matrix
        M_hat = general_estimate_m_matrix_mcar(moments, alpha)

        # ----- Step 4 : debias on D1 -----
        theta_k = lm_mono_debias_estimate_mcar(
            X1, Y1, W11, W21, V1, R1,
            [phi_1, phi_2, phi_3], M_hat)
        cov_k = general_estimate_variance_mcar(moments, alpha)

        theta_list.append(theta_k)
        cov_list.append(cov_k)
        size_list.append(len(D1))

    # ---------- Step 6 : weighted average ----------
    weights = torch.tensor(size_list, device=device, dtype=dtype) / n
    theta_final = sum(w * th for w, th in zip(weights, theta_list))      # (d,)
    cov_final   = sum(w * cv for w, cv in zip(weights, cov_list))        # (d,d)

    return theta_final, cov_final


# ======================================================================
# 2) 2-fold search for alpha*  (Algorithm 2)
# ======================================================================


def lm_mono_debias_budget_constrained_obtain_alpha_mcar(
    X: Tensor, Y: Tensor,
    W1: Tensor, W2: Tensor, V: Tensor,
    R: Tensor,                    # (n,) int64 in {1,2,3}
    tau: float,
    c: float,
    method: str = "mlp",
    eps: float = 1e-6
) -> Tuple[Tensor, float, Tensor]:
    """
    Two-fold cross-fitting (Alg. 2, MCAR) using your Adam optimiser
    `search_alpha_mcar(..., method="adam")` to find the optimal α₁.
    Returns α* = (α₁, α₂, α₃) that minimises Tr(Cov̂(θ̂)).
    """
    # ---------- helper: subset by indices -----------------
    def subset(idxs: Tensor):
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]

    # ---------- deterministic 50/50 split -----------------
    n = X.shape[0]
    device, dtype = X.device, X.dtype
    idx_fold = sample_split(n, 2, device=device)           # returns [idxA, idxB]
    folds = [(idx_fold[0], idx_fold[1]), (idx_fold[1], idx_fold[0])]

    trace_funcs: List = []
    cov_funcs:   List = []

    # ============== two-fold cross-fit loop ===============
    for D_moment, D_build in folds:
        # ----- data partitions ----------------------------
        X1, Y1, W11, W21, V1, R1 = subset(D_moment)   # moment set
        X2, Y2, W12, W22, V2, R2 = subset(D_build)    # ψ/φ set
        # ----- build ψ / φ on D_build ∩ {R=1} ------------
        mask2 = (R2 == 1).squeeze(1)
        theta_pre = lm_fit_wls(X2[mask2], Y2[mask2])

        psi_1, psi_2, psi_3 = lm_build_all_psi(
            X2[mask2], Y2[mask2],
            W12[mask2], W22[mask2], V2[mask2],
            theta_pre, method=method
        )

        phi_1, phi_2, phi_3 = general_build_all_phi_function(psi_2, psi_3)

        # ----- moment function on D_moment ∩ {R=1} -------
        mask1 = (R1 == 1).squeeze(1)
        moment_fn = general_estimate_moments_function_mcar(
            psi_1, phi_1, phi_2, phi_3,
            X1[mask1], Y1[mask1],
            W11[mask1], W21[mask1], V1[mask1]
        )

        # ----- trace / covariance evaluators --------------
        trace_f = general_get_trace_variance_function_alpha_mcar(
            moment_fn, tau, c, return_full=False)
        cov_f   = general_get_trace_variance_function_alpha_mcar(
            moment_fn, tau, c, return_full=True)
        trace_funcs.append(trace_f)
        cov_funcs.append(cov_f)

    # ---------- optimise α₁ with your Adam search ---------
    alpha1_opt = search_alpha_mcar(
        trace_funcs, tau, c,
        eps=eps,
        method="adam",
        device=device
    )
    

    # ---------- construct full α vector ------------------
    alpha2_opt = tau - c * alpha1_opt
    alpha3_opt = 1.0 + (c - 1.0) * alpha1_opt - tau
    alpha_opt = torch.tensor(
        [alpha1_opt, alpha2_opt, alpha3_opt],
        device=device, dtype=dtype
    )                               # (3,)

    # ---------- final covariance & trace -----------------
    cov_opt = sum(f(alpha1_opt) for f in cov_funcs) / len(cov_funcs)
    trace_opt = float(torch.trace(cov_opt).item())

    return alpha_opt, trace_opt, cov_opt

def lm_mono_debias_budget_constrained_obtain_alpha_mcar_cov00(
    X: Tensor, Y: Tensor,
    W1: Tensor, W2: Tensor, V: Tensor,
    R: Tensor,                    # (n,) int64 in {1,2,3}
    tau: float,
    c: float,
    method: str = "mlp",
    eps: float = 1e-6
) -> Tuple[Tensor, float, Tensor]:
    """
    Two-fold cross-fitting (Alg. 2, MCAR) using your Adam optimiser
    `search_alpha_mcar(..., method="adam")` to find the optimal α₁.
    Returns α* = (α₁, α₂, α₃) that minimises Tr(Cov̂(θ̂)).
    """
    # ---------- helper: subset by indices -----------------
    def subset(idxs: Tensor):
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]

    # ---------- deterministic 50/50 split -----------------
    n = X.shape[0]
    device, dtype = X.device, X.dtype
    idx_fold = sample_split(n, 2, device=device)           # returns [idxA, idxB]
    folds = [(idx_fold[0], idx_fold[1]), (idx_fold[1], idx_fold[0])]

    cov00_funcs: List = []

    # ============== two-fold cross-fit loop ===============
    for D_moment, D_build in folds:
        # ----- data partitions ----------------------------
        X1, Y1, W11, W21, V1, R1 = subset(D_moment)   # moment set
        X2, Y2, W12, W22, V2, R2 = subset(D_build)    # ψ/φ set
        # ----- build ψ / φ on D_build ∩ {R=1} ------------
        mask2 = (R2 == 1).squeeze(1)
        theta_pre = lm_fit_wls(X2[mask2], Y2[mask2])

        psi_1, psi_2, psi_3 = lm_build_all_psi(
            X2[mask2], Y2[mask2],
            W12[mask2], W22[mask2], V2[mask2],
            theta_pre, method=method
        )

        phi_1, phi_2, phi_3 = general_build_all_phi_function(psi_2, psi_3)

        # ----- moment function on D_moment ∩ {R=1} -------
        mask1 = (R1 == 1).squeeze(1)
        moment_fn = general_estimate_moments_function_mcar(
            psi_1, phi_1, phi_2, phi_3,
            X1[mask1], Y1[mask1],
            W11[mask1], W21[mask1], V1[mask1]
        )

        # ----- trace / covariance evaluators --------------
        cov00_f = general_get_cov00_function_alpha_mcar(moment_fn, tau, c)
        cov00_funcs.append(cov00_f)

    # ---------- optimise α₁ with your Adam search ---------
    alpha1_opt = search_alpha_mcar(
        cov00_funcs, tau, c,
        eps=eps,
        method="adam",
        device=device
    )
    

    # ---------- construct full α vector ------------------
    alpha2_opt = tau - c * alpha1_opt
    alpha3_opt = 1.0 + (c - 1.0) * alpha1_opt - tau
    alpha_opt = torch.tensor(
        [alpha1_opt, alpha2_opt, alpha3_opt],
        device=device, dtype=dtype
    )                               # (3,)

    # ---------- final covariance & trace -----------------
    cov_f = general_get_trace_variance_function_alpha_mcar(
        moment_fn, tau, c, return_full=True
    )
    cov_opt = cov_f(alpha1_opt)
    cov00 = float(cov_opt[0, 0].item())

    return alpha_opt, cov00, cov_opt

# ======================================================================
# 3) Algorithm 1 wrapper  (choose alpha* & evaluate)
# ======================================================================

def lm_mcar(
    *,
    # --------------- batch sizes -----------------
    n1: int,
    n2: int,
    reps: int,
    # --------------- model dims ------------------
    d_x: int,
    d_u1: int,
    d_u2: int,
    # --------------- model parameters ------------
    theta_star: torch.Tensor,        # (d,)
    beta1_star: torch.Tensor,
    beta2_star: torch.Tensor,
    # --------------- noise -----------------------
    sigma_eps: float,
    # --------------- MCAR / CI -------------------
    alpha_level: float,
    tau: float,
    c: float,
    alpha_init: torch.Tensor,        # (3,)
    # --------------- misc ------------------------
    seed: int = 42,
) -> Dict[str, float | torch.Tensor]:

    device, dtype = theta_star.device, theta_star.dtype
    set_global_seed(seed)

    # ---------- Stage-1 : obtain α* -------------------------
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

    alpha_opt, cov00_opt, _ = lm_mono_debias_budget_constrained_obtain_alpha_mcar_cov00(
        X1, Y1, W1_1, W2_1, V1, R1,
        tau=tau, c=c, method="mlp"
    )

    # baseline α (α₂ = 0)
    alpha1_base  = tau / c
    alpha_baseline = torch.tensor(
        [alpha1_base,
         0.0,
         1.0 + (c - 1.0) * alpha1_base - tau],
        device=device, dtype=dtype
    )

    # ------------- containers for stage-2 -------------------
    err_opt, err_base, err_ols = [], [], []
    len_opt, len_base, len_ols = [], [], []
    cov_opt, cov_base, cov_ols = [], [], []

    # =================== Stage-2 loop =======================
    for rep in tqdm(range(reps),total=reps):
        set_global_seed(seed + rep)

        # ----- Batch-A : α_opt -----------------------------------
        X2A, Y2A, W1A, W2A, V2A, R2A = lm_generate_obs_data_mcar(
            n=n2, d_x=d_x, d_u1=d_u1, d_u2=d_u2,
            theta_star=theta_star,
            beta1_star=beta1_star,
            beta2_star=beta2_star,
            alpha=alpha_opt,
            Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
            sigma_eps=sigma_eps,
        )

        theta_opt_vec, cov_opt_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2A, Y2A, W1A, W2A, V2A, R2A,
            alpha=alpha_opt, method="mlp"
        )

        se_opt_1 = torch.sqrt(cov_opt_mat[0, 0] / n2).item()
        ci_opt_low, ci_opt_high = wald_ci(
            mu_hat=theta_opt_vec[0].item(),
            se=se_opt_1,
            alpha_level=alpha_level
        )

        # ----- Batch-B : α_baseline -----------------------------
        X2B, Y2B, W1B, W2B, V2B, R2B = lm_generate_obs_data_mcar(
            n=n2, d_x=d_x, d_u1=d_u1, d_u2=d_u2,
            theta_star=theta_star,
            beta1_star=beta1_star,
            beta2_star=beta2_star,
            alpha=alpha_baseline,
            Sigma_X=None, Sigma_U1=None, Sigma_U2=None,
            sigma_eps=sigma_eps,
        )

        theta_base_vec, cov_base_mat = lm_mono_debias_estimate_mcar_crossfit(
            X2B, Y2B, W1B, W2B, V2B, R2B,
            alpha=alpha_baseline, method="mlp"
        )

        se_base_1 = torch.sqrt(cov_base_mat[0, 0] / n2).item()
        ci_base_low, ci_base_high = wald_ci(
            mu_hat=theta_base_vec[0].item(),
            se=se_base_1,
            alpha_level=alpha_level
        )

        # ----- OLS reference ------------------------------------
        mask_ols  = ~torch.isnan(Y2B).view(-1)
        X_ols = X2B[mask_ols]
        Y_ols = Y2B[mask_ols]
        n_ols = X_ols.shape[0]
        d_ols = X_ols.shape[1]
        theta_ols_vec = lm_fit_ols(X_ols, Y_ols)

        # Estimate covariance for OLS
        residuals = Y_ols - (X_ols @ theta_ols_vec.view(-1, 1))
        s_squared = (residuals.T @ residuals).item() / (n_ols - d_ols)
        inv_xtx = torch.inverse(X_ols.T @ X_ols)
        
        se_ols_1 = torch.sqrt(s_squared * inv_xtx[0, 0]).item()
        ci_ols_low, ci_ols_high = wald_ci(
            mu_hat=theta_ols_vec[0].item(),
            se=se_ols_1,
            alpha_level=alpha_level
        )

        # ----- accumulate metrics ------------------------------
        err_opt .append(torch.linalg.norm(theta_opt_vec  - theta_star).item())
        err_base.append(torch.linalg.norm(theta_base_vec - theta_star).item())
        err_ols .append(torch.linalg.norm(theta_ols_vec  - theta_star).item())

        len_opt .append(ci_opt_high  - ci_opt_low)
        len_base.append(ci_base_high - ci_base_low)
        len_ols.append(ci_ols_high - ci_ols_low)

        cov_opt .append(int(ci_opt_low  <= theta_star[0].item() <= ci_opt_high))
        cov_base.append(int(ci_base_low <= theta_star[0].item() <= ci_base_high))
        cov_ols.append(int(ci_ols_low <= theta_star[0].item() <= ci_ols_high))

    # ---------------- aggregate & return ---------------------
    return dict(
        alpha_opt = ((alpha_opt.cpu() * 1e4).round() / 1e4).tolist(),
        cov00_opt = float(cov00_opt),
        mean_l2_opt = float(sum(err_opt)  / reps),
        mean_l2_base= float(sum(err_base) / reps),
        mean_l2_ols = float(sum(err_ols)  / reps),
        mean_len_opt = float(sum(len_opt)  / reps),
        mean_len_base= float(sum(len_base) / reps),
        mean_len_ols = float(sum(len_ols) / reps),
        covg_opt  = float(sum(cov_opt)  / reps),
        covg_base = float(sum(cov_base) / reps),
        covg_ols = float(sum(cov_ols) / reps),
    )
