# =============================================================================
# estimators.py  –  Statistical objects for MCAR mono-debias (pure Torch)
#
# Table of Contents (only the functions shown in the screenshot, in order)
# -----------------------------------------------------------------------------
# 0) Ordinary Least Squares (OLS) & Weighted Least Squares (WLS)
#    lm_fit_ols(...)                            # OLS regression
#    lm_fit_wls(...)                            # WLS regression
# 1) ψ-models
#    lm_build_all_psi(...)                      # compute ψ-values (no gradients)
#
# 2) φ-functions / phi-functions
#    general_build_all_phi(...)                 # evaluate φ on a fixed alpha
#    general_build_all_phi_function(...)        # return callables φ1/φ2/φ3
#
# 3) Empirical moments
#    general_estimate_moments_mcar(...)         # E[ψψᵀ], E[φ_jφ_jᵀ], Cov(ψ,φ_j)
#    general_estimate_moments_function_mcar(...)# return moment_fn(alpha_vec)
#
# 4) M-matrix & variance
#    general_estimate_m_matrix_mcar(...)        # estimate M^(1)
#    general_estimate_variance_mcar(...)        # estimate Cov(θ̂)
#
# 5) g(alpha1) under budget constraint
#    general_get_trace_variance_function_alpha_mcar(...)  # return g(α1)
#
# 6) One-dimensional search for α1
#    _golden_section(...)                       # derivative-free golden section
#    _adam_section(...)                         # gradient-based Adam search
#    search_alpha_mcar(...)                     # public API: choose "golden"/"adam"
#
# 7) Efficient estimator (Algorithm 3, MCAR)
#    lm_mono_debias_estimate_mcar(...)          # 3-fold cross-fit efficient θ̂
# =============================================================================


from __future__ import annotations
from typing import Dict, Sequence, Tuple, List, Callable, Literal, Optional, Union
import torch
from torch import Tensor


from .models  import train_model

# ============================================================
# Ordinary Least Squares (OLS) – Torch version
# ============================================================

def lm_fit_ols(X: Tensor, Y: Tensor) -> Tensor:
    """
    Fit an Ordinary Least Squares (OLS) regression model.

    Parameters
    ----------
    X : Tensor (n_samples, n_features)
        Design matrix.
    Y : Tensor (n_samples, 1)
        Response vector.

    Returns
    -------
    coeff : Tensor (n_features,)
        Estimated OLS coefficients.
    """
    # Use least-squares solver (stable on CPU / CUDA)
    #   θ̂ = (XᵀX)⁻¹ XᵀY   is equivalent to   lstsq(X, Y)
    sol = torch.linalg.lstsq(X, Y).solution   # (n_features, 1)
    return sol.squeeze(1)                                  # (n_features,)


# ============================================================
# Weighted Least Squares (WLS) – Torch version
# ============================================================

def lm_fit_wls(
    X: Tensor,
    Y: Tensor,
    w: Optional[Tensor] = None
) -> Tensor:
    """
    Fit a Weighted Least Squares (WLS) regression model.

    Parameters
    ----------
    X : Tensor (n_samples, n_features)
        Design matrix.
    Y : Tensor (n_samples, 1)
        Response vector (must be a column vector).
    w : Tensor (n_samples,), optional
        Sample weights. If None or all ones, falls back to OLS.

    Returns
    -------
    coeff : Tensor (n_features,)
        Estimated WLS coefficients.
    """
    if w is None or torch.allclose(w, torch.ones_like(Y.squeeze())):
        return lm_fit_ols(X, Y)

    # Re-scale rows by sqrt(weights):  X̃ = diag(√w) · X,  Ỹ = √w · Y
    sqrt_w = torch.sqrt(w).unsqueeze(1)        # (n_samples, 1)
    Xw = X * sqrt_w                            # (n_samples, n_features)
    Yw = Y * sqrt_w                            # (n_samples, 1)

    sol = torch.linalg.lstsq(Xw, Yw).solution  # (n_features, 1)
    return sol.squeeze(1)                                    # (n_features,)

# ----------------------------------------------------------------------
# 1) ψ-functions   ------------------------------------------------------
# ----------------------------------------------------------------------

def lm_build_all_psi(
    X:  Tensor,
    Y:  Tensor,
    W1: Tensor,
    W2: Tensor,
    V:  Tensor,
    theta: Tensor,
    *,
    method: str = "mlp",
) -> Tuple[
    Callable[[Tensor, Tensor],                   Tensor],   # ψ₁
    Callable[[Tensor, Tensor, Tensor, Tensor],   Tensor],   # ψ₂
    Callable[[Tensor, Tensor, Tensor],           Tensor],   # ψ₃
]:
    """
    Returns ψ₁, ψ₂, ψ₃ that accept batches and return (n_batch, d) corrections.
    All inputs must be float32 and on the same device.
    """
    # ------------------------------------------------------------------
    # Reshape inputs to (n, 1) where necessary
    # ------------------------------------------------------------------
    Y  = Y.view(-1, 1)
    W1 = W1.view(-1, 1)
    W2 = W2.view(-1, 1)
    V  = V.view(-1, 1)
    theta = theta.view(-1, 1)

    # ------------------------------------------------------------------
    # Dimensions and precomputed inverse
    # ------------------------------------------------------------------
    n, d = X.shape
    inv_exx = torch.inverse((X.T @ X) / n)  # (d, d)

    # ------------------------------------------------------------------
    # Train auxiliary models for ψ₂ and ψ₃
    # ------------------------------------------------------------------
    feats_psi2 = torch.cat([X, W1, W2, V], dim=1)
    feats_psi3 = torch.cat([X, W1, W2], dim=1)

    with torch.enable_grad():
        psi_model_2 = train_model(feats_psi2, Y, method=method).eval()
        psi_model_3 = train_model(feats_psi3, Y, method=method).eval()

    # ------------------------------------------------------------------
    # Core helper: ψ_core for batch input
    # ------------------------------------------------------------------
    def _psi_core_batch(Xv: Tensor, Y_like: Tensor) -> Tensor:
        """
        Xv:      (n_val, d)
        Y_like:  (n_val, 1)
        Returns: (n_val, d)
        """
        resid = Y_like - Xv @ theta                   # (n_val, 1)
        psi_batch = (Xv.unsqueeze(2) * resid.unsqueeze(1))  # (n_val, d, 1)
        return torch.matmul(psi_batch.squeeze(2), inv_exx.T)  # (n_val, d)

    # ------------------------------------------------------------------
    # ψ functions that support batch input
    # ------------------------------------------------------------------
    def psi_1(X_val: Tensor, Y_val: Tensor) -> Tensor:
        Xv = X_val.view(-1, d)
        Yv = Y_val.view(-1, 1)
        return _psi_core_batch(Xv, Yv)  # (n_val, d)

    def psi_2(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        Xv = X_val.view(-1, d)
        feats = torch.cat([
            Xv,
            W1_val.view(-1, 1),
            W2_val.view(-1, 1),
            V_val.view(-1, 1)
        ], dim=1)
        Y_hat = psi_model_2(feats).view(-1, 1)   # (n_val, 1)
        return _psi_core_batch(Xv, Y_hat)        # (n_val, d)

    def psi_3(X_val: Tensor, W1_val: Tensor, W2_val: Tensor) -> Tensor:
        Xv = X_val.view(-1, d)
        feats = torch.cat([
            Xv,
            W1_val.view(-1, 1),
            W2_val.view(-1, 1)
        ], dim=1)
        Y_hat = psi_model_3(feats).view(-1, 1)   # (n_val, 1)
        return _psi_core_batch(Xv, Y_hat)        # (n_val, d)

    return psi_1, psi_2, psi_3

# ----------------------------------------------------------------------
# 2) φ-functions / phi-functions
# ----------------------------------------------------------------------

def general_build_all_phi(
    psi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    psi_3: Callable[[Tensor, Tensor, Tensor], Tensor],
    alpha: Tensor,  # (3,) Tensor: [alpha1, alpha2, alpha3]
) -> Tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
]:
    """
    Construct φ₁, φ₂, φ₃ from ψ₂, ψ₃ and fixed scalar weights α = [α₁, α₂, α₃].

    Each φ_j has signature:
        phi_j(X_val, W1_val, W2_val, V_val) -> Tensor of shape (n, d)
    """

    def phi_1(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        n = X_val.shape[0]
        a1 = alpha[0].expand(n, 1)
        a2 = alpha[1].expand(n, 1)
        a3 = alpha[2].expand(n, 1)

        denom_sum = a1 + a2
        denom_prod = denom_sum * a1

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        term1 = (a2 / denom_prod) * psi2_val
        term2 = (a3 / denom_sum) * psi3_val

        return -term1 - term2

    def phi_2(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        n = X_val.shape[0]
        a1 = alpha[0].expand(n, 1)
        a2 = alpha[1].expand(n, 1)
        a3 = alpha[2].expand(n, 1)

        denom_sum = a1 + a2

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return (1.0 / denom_sum) * psi2_val - (a3 / denom_sum) * psi3_val

    def phi_3(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        return psi_3(X_val, W1_val, W2_val)

    return phi_1, phi_2, phi_3


def general_build_all_phi_mar(
    psi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    psi_3: Callable[[Tensor, Tensor, Tensor], Tensor],
    alpha_fn: Callable[[Tensor, Tensor, Tensor], Tensor],  # returns (n, 3)
) -> Tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
]:
    """
    Construct φ₁, φ₂, φ₃ from ψ₂, ψ₃ and alpha_fn producing weights [α₁, α₂, α₃].

    Each φ_j has signature:
        phi_j(X_val, W1_val, W2_val, V_val) -> Tensor of shape (n, d)

    All inputs and outputs are torch.Tensor and assumed on the same device.
    """

    def phi_1(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        alpha = alpha_fn(X_val, W1_val, W2_val)  # (n, 3)
        a1, a2, a3 = alpha[:, 0:1], alpha[:, 1:2], alpha[:, 2:3]  # shape (n, 1)

        denom_sum = a1 + a2
        denom_prod = denom_sum * a1

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return - (a2 / denom_prod) * psi2_val - (a3 / denom_sum) * psi3_val

    def phi_2(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        alpha = alpha_fn(X_val, W1_val, W2_val)
        a1, a2, a3 = alpha[:, 0:1], alpha[:, 1:2], alpha[:, 2:3]

        denom_sum = a1 + a2

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return (1.0 / denom_sum) * psi2_val - (a3 / denom_sum) * psi3_val

    def phi_3(X_val: Tensor, W1_val: Tensor, W2_val: Tensor, V_val: Tensor) -> Tensor:
        return psi_3(X_val, W1_val, W2_val)

    return phi_1, phi_2, phi_3



def general_build_all_phi_function(
    psi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    psi_3: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tuple[
    Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
]:
    """
    Return three φ-functions for MCAR debiasing, fully in PyTorch.

    Each phi_j has signature
        phi_j(alpha, X_val, W1_val, W2_val, V_val) -> Tensor shape (d,)

    All inputs and outputs are torch.Tensor.
    """
    def phi_1(
        alpha: Tensor,
        X_val: Tensor,
        W1_val: Tensor,
        W2_val: Tensor,
        V_val: Tensor,
    ) -> Tensor:
        # alpha: (3,)
        alpha1, alpha2, alpha3 = alpha.unbind(0)
        denom_sum  = alpha1 + alpha2
        denom_prod = denom_sum * alpha1

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return -(alpha2 / denom_prod) * psi2_val \
               - (alpha3 / denom_sum) * psi3_val

    def phi_2(
        alpha: Tensor,
        X_val: Tensor,
        W1_val: Tensor,
        W2_val: Tensor,
        V_val: Tensor,
    ) -> Tensor:
        alpha1, alpha2, alpha3 = alpha.unbind(0)
        denom_sum = alpha1 + alpha2

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return (1.0 / denom_sum) * psi2_val \
               - (alpha3 / denom_sum) * psi3_val

    def phi_3(
        alpha: Tensor,
        X_val: Tensor,
        W1_val: Tensor,
        W2_val: Tensor,
        V_val: Tensor,
    ) -> Tensor:
        # phi_3 = ψ₃, independent of alpha and V
        return psi_3(X_val, W1_val, W2_val)

    return phi_1, phi_2, phi_3

# ----------------------------------------------------------------------
# 3) empirical moments
# ----------------------------------------------------------------------

def general_estimate_moments_mcar(
    psi_1: Callable[[Tensor, Tensor], Tensor],
    phi_1: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_2: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_3: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    X: Tensor,
    Y: Tensor,
    W1: Tensor,
    W2: Tensor,
    V: Tensor,
) -> Dict[str, List[Tensor] | Tensor]:
    """
    Estimate second-order moments and cross-moments under MCAR.

    Assumptions
    -----------
    * All input tensors are on the same device and have the same dtype.
    * psi_1, phi_1, phi_2, phi_3 are batch functions:
        - psi_1(X, Y)             -> (n, d)
        - phi_j(X, W1, W2, V)     -> (n, d)

    Returns
    -------
    dict
        {
          "E[psi1 psi1^T]" : (d, d) Tensor,
          "E[phi_j phi_j^T]": list[(d, d)],
          "Cov(psi1, phi_j)": list[(d, d)],
        }
    """
    n, d = X.shape

    # 1. Evaluate basis functions for the entire batch
    psi1_vals = psi_1(X, Y)                 # (n, d)
    phi1_vals = phi_1(X, W1, W2, V)         # (n, d)
    phi2_vals = phi_2(X, W1, W2, V)         # (n, d)
    phi3_vals = phi_3(X, W1, W2, V)         # (n, d)

    # Sanity checks (optional)
    assert psi1_vals.shape == (n, d), "psi_1 must return shape (n, d)"
    for idx, phi in enumerate([phi1_vals, phi2_vals, phi3_vals], 1):
        assert phi.shape == (n, d), f"phi_{idx} must return shape (n, d)"

    # 2. E[ψ₁ ψ₁ᵀ]
    E_psi1_psi1T = (psi1_vals.T @ psi1_vals) / n  # (d, d)

    # 3. E[φⱼ φⱼᵀ] and Cov(ψ₁, φⱼ)
    phi_vals = [phi1_vals, phi2_vals, phi3_vals]
    E_phi_phiT_list: List[Tensor] = []
    Cov_psi_phi_list: List[Tensor] = []

    for phi_j in phi_vals:
        E_phi_phiT_list.append((phi_j.T @ phi_j) / n)      # (d, d)
        Cov_psi_phi_list.append((psi1_vals.T @ phi_j) / n) # (d, d)

    # 4. Pack results
    return {
        "E[psi1 psi1^T]":   E_psi1_psi1T,
        "E[phi_j phi_j^T]": E_phi_phiT_list,
        "Cov(psi1, phi_j)": Cov_psi_phi_list,
    }


def general_estimate_moments_function_mcar(
    psi_1: Callable[[Tensor, Tensor], Tensor],
    phi_1: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_2: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    phi_3: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    X: Tensor,
    Y: Tensor,
    W1: Tensor,
    W2: Tensor,
    V: Tensor,
) -> Callable[[Tensor], Dict[str, List[Tensor] | Tensor]]:
    """
    Construct a function moment_fn(alpha) that computes empirical moment matrices 
    under the MCAR setting, based on precomputed psi_1 and on-demand phi_j(alpha).

    Assumptions
    -----------
    - psi_1 is batch-aware:      psi_1(X, Y) -> (n, d)
    - phi_j is batch-aware:      phi_j(alpha, X, W1, W2, V) -> (n, d)
    - All tensors share the same device and dtype.
    """
    n, d = X.shape
    # --------------------------------------------------------
    # 1) Pre-compute ψ₁ values (independent of alpha) --------
    # --------------------------------------------------------
    with torch.no_grad():
        psi1_vals: Tensor       = psi_1(X, Y)                      # (n, d)
        E_psi1_psi1T: Tensor    = (psi1_vals.T @ psi1_vals) / n    # (d, d)

    # --------------------------------------------------------
    # 2) Build and return moment_fn --------------------------
    # --------------------------------------------------------
    def moment_fn(alpha: Tensor) -> Dict[str, List[Tensor] | Tensor]:
        # Compute phi_j(alpha) for j = 1,2,3
        phi_vals = [
            phi_func(alpha, X, W1, W2, V)
            for phi_func in (phi_1, phi_2, phi_3)
        ]

        # Compute E[phi_j phi_j^T] and Cov(psi1, phi_j)
        E_phi_phiT_list = [(phi.T @ phi) / n for phi in phi_vals]
        Cov_psi_phi_list = [(psi1_vals.T @ phi) / n for phi in phi_vals]

        return {
            "E[psi1 psi1^T]":   E_psi1_psi1T,
            "E[phi_j phi_j^T]": E_phi_phiT_list,
            "Cov(psi1, phi_j)": Cov_psi_phi_list,
        }

    return moment_fn

# ----------------------------------------------------------------------
# 4) M-matrix & variance
# ----------------------------------------------------------------------

def general_estimate_m_matrix_mcar(
    moment_dict: Dict[str, List[Tensor] | Tensor],
    alphas: Sequence[float | Tensor],
) -> Tensor:
    """
    Estimate M^{(1)} under the MCAR assumption.

    Formula
    -------
        M = - Cov(ψ₁, φ₁)  @  [ Σⱼ αⱼ E[φⱼ φⱼᵀ] ]^{-1}

    Parameters
    ----------
    moment_dict : dict
        {
          "Cov(psi1, phi_j)" : list[(d, d)]  # we use only index 0 (φ₁)
          "E[phi_j phi_j^T]" : list[(d, d)]  # length = 3
        }
        All entries are already `torch.Tensor`s.
    alphas : length-3 sequence
        Scalar weights (Python floats, ints, or 0-D Tensors).

    Returns
    -------
    Tensor
        (d, d) matrix M on the same device / dtype as the inputs.
    """
    # 1. Extract required tensors
    cov_psi1_phi1: Tensor = moment_dict["Cov(psi1, phi_j)"][0]   # (d, d)
    E_phi_phiT_list: List[Tensor] = moment_dict["E[phi_j phi_j^T]"]

    # 2. Weighted sum Σⱼ αⱼ E[φⱼ φⱼᵀ]
    #    Convert each alpha to the same dtype / device as the moment matrices.
    weighted_sum = sum(
        (torch.as_tensor(alpha, dtype=cov_psi1_phi1.dtype, device=cov_psi1_phi1.device)
         * E_phi) for alpha, E_phi in zip(alphas, E_phi_phiT_list)
    )  # (d, d)

    # 3. Compute M   (torch.linalg.inv handles CPU / CUDA transparently)
    M = -cov_psi1_phi1 @ torch.linalg.inv(weighted_sum)  # (d, d)
    return M



def general_estimate_variance_mcar(
    moment_dict: Dict[str, List[Tensor] | Tensor],
    alphas: Sequence[float | Tensor],
) -> Tensor:
    """
    Estimate  Cov(θ̂)  for the MCAR efficient estimator.

    Formula
    -------
        Cov̂(θ̂) =  Ê[ψ₁ ψ₁ᵀ] / α₁
                   – Cov̂(ψ₁, φ̂₁) · (Σⱼ αⱼ Ê[φ̂ⱼ φ̂ⱼᵀ])⁻¹ · Cov̂(ψ₁, φ̂₁)ᵀ

    Parameters
    ----------
    moment_dict : dict
        Output of `general_estimate_moments_mcar_torch`, containing
          "E[psi1 psi1^T]"   : (d, d) Tensor
          "E[phi_j phi_j^T]" : list[Tensor]  length-3, each (d, d)
          "Cov(psi1, phi_j)" : list[Tensor]  length-3, each (d, d)
    alphas : length-3 sequence
        (alpha1, alpha2, alpha3) – Python scalars or 0-D Tensors.

    Returns
    -------
    Tensor
        (d, d) covariance estimate on the same device / dtype as the inputs.
    """
    # ---------------- 1) unpack moments ---------------------
    E_psi1_psi1T: Tensor        = moment_dict["E[psi1 psi1^T]"]      # (d, d)
    E_phi_phiT_list: List[Tensor] = moment_dict["E[phi_j phi_j^T]"]   # 3 × (d, d)
    Cov_psi_phi_list: List[Tensor] = moment_dict["Cov(psi1, phi_j)"]  # 3 × (d, d)

    # ---------------- 2) prepare alphas ---------------------
    # Cast each alpha to a 0-D tensor on the same device / dtype
    device, dtype = E_psi1_psi1T.device, E_psi1_psi1T.dtype
    alpha1, alpha2, alpha3 = [
        torch.as_tensor(a, device=device, dtype=dtype) for a in alphas
    ]

    # ---------------- 3) Σⱼ αⱼ E[φⱼ φⱼᵀ] -------------------
    weighted_phi_cov = (
        alpha1 * E_phi_phiT_list[0]
        + alpha2 * E_phi_phiT_list[1]
        + alpha3 * E_phi_phiT_list[2]
    )  # (d, d)

    inv_weighted_phi_cov = torch.linalg.inv(weighted_phi_cov)  # (d, d)
    cov_psi_phi1         = Cov_psi_phi_list[0]                # (d, d)

    # ---------------- 4) covariance estimator ---------------
    correction = cov_psi_phi1 @ inv_weighted_phi_cov @ cov_psi_phi1.T  # (d, d)
    cov_theta  = E_psi1_psi1T / alpha1 - correction                    # (d, d)
    return cov_theta

# ----------------------------------------------------------------------
# 5) g(alpha1)  under budget constraint
# ----------------------------------------------------------------------
def general_get_trace_variance_function_alpha_mcar(
    moment_fn: Callable[[Tensor], Dict[str, List[Tensor] | Tensor]],
    tau: float,
    c: float,
    return_full: bool = False,
) -> Callable[[Tensor], Union[Tensor, Tensor]]:
    """
    Return a function g(alpha1) that produces either
        • Tr(  Cov̂(θ̂) )   (default) or
        •       Cov̂(θ̂)    (if return_full=True)
    under the budget constraint (2.3):

        alpha2 = tau - c * alpha1
        alpha3 = 1 + (c - 1) * alpha1 - tau

    Parameters
    ----------
    moment_fn : callable
        Output of `general_estimate_moments_function_mcar` (Torch version).
        It maps a length-3 alpha vector to the required moment dictionary.
    tau : float
        Constraint alpha1 + alpha2 = tau
    c : float
        Constraint parameter in Eq. (2.3)
    return_full : bool, default False
        If True, g(alpha1) returns the full covariance matrix; otherwise its trace.

    Returns
    -------
    g : callable
        g(alpha1) -> scalar (trace) or (d, d) Tensor, depending on `return_full`.
        Accepts a 0-D Tensor for alpha1.
    """

    def g(alpha1: Tensor) -> Tensor:
        """
        Evaluate the covariance (or its trace) at a given alpha1.
        """

        # -------- 1) build full alpha vector under constraint -------------
        alpha_vec = torch.stack([
            alpha1,
            tau - c * alpha1,
            1.0 + (c - 1.0) * alpha1 - tau
        ]).to(alpha1)  # Keep same device and dtype

        # -------- 2) retrieve empirical moments ---------------------------
        moment_dict = moment_fn(alpha_vec)
        E_psi1_psi1T = moment_dict["E[psi1 psi1^T]"]
        E_phi_phiT_lst = moment_dict["E[phi_j phi_j^T]"]
        Cov_psi_phi1 = moment_dict["Cov(psi1, phi_j)"][0]

        # -------- 3) compute covariance estimator -------------------------
        weighted_phi_cov = sum(a * E for a, E in zip(alpha_vec, E_phi_phiT_lst))
        inv_weighted = torch.linalg.inv(weighted_phi_cov)

        cov_theta = E_psi1_psi1T / alpha_vec[0] - Cov_psi_phi1 @ inv_weighted @ Cov_psi_phi1.T
        out = cov_theta if return_full else torch.trace(cov_theta)

        return out
    return g

def general_get_cov00_function_alpha_mcar(
    moment_fn: Callable[[Tensor], Dict[str, List[Tensor] | Tensor]],
    tau: float,
    c: float,
) -> Callable[[Tensor], Tensor]:
    """
    Return a function g(alpha1) that evaluates Cov(θ̂)[0, 0]
    under the budget constraint (2.3).

    This isolates the variance of the first component of the parameter vector.
    """

    def g(alpha1: Tensor) -> Tensor:
        # -------- 1) build full alpha vector -----------------------------
        alpha_vec = torch.stack([
            alpha1,
            tau - c * alpha1,
            1.0 + (c - 1.0) * alpha1 - tau
        ]).to(alpha1)

        # -------- 2) retrieve empirical moments --------------------------
        moment_dict = moment_fn(alpha_vec)
        E_psi1_psi1T = moment_dict["E[psi1 psi1^T]"]
        E_phi_phiT_lst = moment_dict["E[phi_j phi_j^T]"]
        Cov_psi_phi1 = moment_dict["Cov(psi1, phi_j)"][0]

        # -------- 3) compute covariance estimator ------------------------
        weighted_phi_cov = sum(a * E for a, E in zip(alpha_vec, E_phi_phiT_lst))
        inv_weighted = torch.linalg.inv(weighted_phi_cov)

        cov_theta = E_psi1_psi1T / alpha_vec[0] - Cov_psi_phi1 @ inv_weighted @ Cov_psi_phi1.T
        return cov_theta[0, 0]  # return variance of θ̂₀

    return g


# ---------------------------------------------------------------------
# 6) Derivative-free: Golden-section search  (kept unchanged)
# ---------------------------------------------------------------------
def _golden_section(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> float:
    """Unimodal 1-D minimisation via golden-section."""
    phi = 0.5 * (3.0 - 5 ** 0.5)          # 1 − Φ
    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    f1, f2 = f(x1), f(x2)

    for _ in range(max_iter):
        if abs(hi - lo) < tol:
            break
        if f1 > f2:                       # keep [lo, x2]
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = f(x2)
        else:                             # keep [x1, hi]
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = f(x1)

    return (lo + hi) / 2.0


# ---------------------------------------------------------------------
# 6) Gradient-based: Adam search (1-D, closed interval)
# ---------------------------------------------------------------------
def _adam_section(
    f: Callable[[Tensor], Tensor],
    lo: float,
    hi: float,
    *,
    lr: float = 0.05,
    iters: int = 200,
    device: torch.device | None = None,
) -> float:
    """
    One-dimensional scalar optimisation using Adam.

    We optimise an unconstrained real variable z, and map it to
        alpha1 = lo + sigmoid(z) * (hi − lo)  ∈ (lo, hi)
    to respect the box constraints.
    """
    device = torch.device("cpu") if device is None else device

    # Unconstrained parameter (requires_grad=True for autograd)
    z = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
    optimiser = torch.optim.Adam([z], lr=lr)

    lo_t = torch.tensor(lo, device=device)
    hi_t = torch.tensor(hi, device=device)

    def _to_alpha(z_val: Tensor) -> Tensor:
        """Map ℝ → (lo, hi) via sigmoid."""
        return lo_t + torch.sigmoid(z_val) * (hi_t - lo_t)

    # Enable gradients only inside this scope
    with torch.enable_grad():
        for _ in range(iters):
            optimiser.zero_grad(set_to_none=True)
            alpha1 = _to_alpha(z)
            loss = f(alpha1)             # f returns scalar Tensor
            loss.backward()
            optimiser.step()

    # Detach to float for downstream use
    return _to_alpha(z).detach()


# ---------------------------------------------------------------------
# 6) Public API: search_alpha_mcar  (choose "golden" | "adam")
# ---------------------------------------------------------------------
def search_alpha_mcar(
    trace_funcs: Sequence[Callable[[Tensor], Tensor]],
    tau: float,
    c: float,
    *,
    eps: float = 1e-6,
    method: str = "adam",              # "golden" | "adam"
    device: torch.device | None = None,
) -> float:
    """
    Minimise the averaged trace of covariance matrices over alpha1.

    The feasible set for alpha1 is an interval [lo, hi] derived from
    the budget and linear constraints:
        alpha2 = tau − c * alpha1
        alpha3 = 1 + (c − 1) * alpha1 − tau
    """
    up1  = tau / c
    low1 = 0.0 if c == 1 else (tau - 1) / (c - 1)
    lo, hi = max(eps, low1), min(1.0 - eps, up1)

    # Callable that returns mean trace( Σ_hat ) for a given alpha1
    def mean_trace(a1: Tensor) -> Tensor:
        if not isinstance(a1, Tensor):
            a1 = torch.tensor(a1, dtype=torch.float32, device=device)

        val = sum(f(a1) for f in trace_funcs) / len(trace_funcs)
        return val
    if method == "adam":
        return _adam_section(mean_trace, lo, hi, device=device)
    elif method == "golden":
        # _golden_section expects a function that takes a float
        def mean_trace_float(a1_float: float) -> float:
            return float(mean_trace(torch.tensor(a1_float)).detach())
        return _golden_section(mean_trace_float, lo, hi)
    else:
        raise ValueError(f"unknown method '{method}', choose 'golden' or 'adam'")
    

# ----------------------------------------------------------------------
def lm_mono_debias_estimate_mcar(
    X: Tensor,                   # (n, d)
    Y: Tensor,                   # (n,)
    W1: Tensor,                  # (n,)
    W2: Tensor,                  # (n,)
    V: Tensor,                   # (n,)
    R: Tensor,                   # (n,)  int64 values in {1,2,3}
    phi_list: List[Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]],
    M_hat: Tensor,               # (d, d)
    w_init: Optional[Tensor] = None  # (n,) or None
) -> Tensor:
    """
    Efficient estimator θ̂^(1) under MCAR (mono-debiasing), Torch version.

    Each φ_j in `phi_list` must accept (X_batch, W1_batch, W2_batch, V_batch)
    and return a (batch_size, d) tensor on the same device / dtype.
    """
    n, d = X.shape
    device, dtype = X.device, X.dtype

    # ------------------------------------------------------------------
    # Step 1: initial estimator θ̃ via (weighted) least squares
    # ------------------------------------------------------------------
    Y_flat = Y.view(-1)
    mask_obs = ~torch.isnan(Y_flat)                 # (n,)
    X_obs = X[mask_obs]                             # (n_obs, d)
    Y_obs = Y_flat[mask_obs]                        # (n_obs,)

    if w_init is not None:
        w_obs = w_init[mask_obs]                    # (n_obs,)
        sqrt_w = torch.sqrt(w_obs).unsqueeze(1)     # (n_obs, 1)
        Xw = X_obs * sqrt_w                         # weighted X
        Yw = Y_obs * sqrt_w.squeeze()               # (n_obs,)
    else:
        Xw, Yw = X_obs, Y_obs

    theta_tilde = torch.linalg.lstsq(Xw, Yw).solution.view(-1)  # (d,)

    # ------------------------------------------------------------------
    # Step 2: correction term  (1/n) Σ_i φ_{r_i}
    # ------------------------------------------------------------------
    mask_r1 = (R == 1).view(-1)
    mask_r2 = (R == 2).view(-1)
    mask_r3 = (R == 3).view(-1)

    phi1_vals = phi_list[0](X[mask_r1], W1[mask_r1], W2[mask_r1], V[mask_r1]) if mask_r1.any() else torch.zeros((0, d), device=device)
    phi2_vals = phi_list[1](X[mask_r2], W1[mask_r2], W2[mask_r2], V[mask_r2]) if mask_r2.any() else torch.zeros((0, d), device=device)
    phi3_vals = phi_list[2](X[mask_r3], W1[mask_r3], W2[mask_r3], V[mask_r3]) if mask_r3.any() else torch.zeros((0, d), device=device)

    correction = torch.cat([phi1_vals, phi2_vals, phi3_vals], dim=0).mean(dim=0)  # (d,)

    # ------------------------------------------------------------------
    # Step 3: final debiased estimator
    # ------------------------------------------------------------------
    theta_hat = theta_tilde + M_hat @ correction  # (d,)

    return theta_hat
