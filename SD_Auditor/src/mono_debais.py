import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS, WLS
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.neural_network import MLPRegressor, MLPClassifier
from scipy.optimize import minimize_scalar, OptimizeResult

import sys, os
from src.utils import sample_split_three, sample_split_two



###############################
def lm_fit_ols(X, Y):
    """
    Fit an Ordinary Least Squares (OLS) regression model.

    Args:
        X: Design matrix (numpy.ndarray), shape (n_samples, n_features).
        Y: Response vector (numpy.ndarray), shape (n_samples,).

    Returns:
        Estimated OLS coefficients (numpy.ndarray), shape (n_features,).
    """
    model = OLS(Y, exog=X).fit()
    return model.params



###############################
def lm_fit_wls(X, Y, w=None):
    """
    Fit a Weighted Least Squares (WLS) regression model.

    Args:
        X: Design matrix (numpy.ndarray), shape (n_samples, n_features).
        Y: Response vector (numpy.ndarray), shape (n_samples,).
        w: Sample weights (numpy.ndarray, optional), shape (n_samples,). 
           If None or all ones, falls back to OLS.

    Returns:
        Estimated WLS coefficients (numpy.ndarray), shape (n_features,).
    """
    if w is None or np.all(w == 1):
        return lm_fit_ols(X, Y)

    model = WLS(Y, exog=X, weights=w).fit()
    return model.params



###############################
def lm_fit_y_general(features, Y, method='mlp'):
    """
    Fit a regression or classification model to predict Y from input features.

    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        Y (np.ndarray): Target variable of shape (n_samples,).
        method (str): Model type to use. Options:
            - 'mlp'        : MLPRegressor
            - 'mlpclass'   : MLPClassifier
            - 'linreg'     : LinearRegression
            - 'logistic'   : LogisticRegression
            - 'xgboost'    : XGBRegressor
            - 'tree'       : DecisionTreeRegressor
        seed (int): Random seed for reproducibility. Defaults to global SEED.

    Returns:
        predict_fn (Callable): Function that takes new features and returns predicted Y.
    """
    if method == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(64, 64, 32), max_iter=1000, learning_rate="adaptive", learning_rate_init=1e-3)
    elif method == 'mlpclass':
        model = MLPClassifier(hidden_layer_sizes=(32, 32, 32), max_iter=600)
    elif method == 'linreg':
        model = LinearRegression()
    elif method == 'logistic':
        model = LogisticRegression(max_iter=600)
    elif method == 'xgboost':
        model = xgb.XGBRegressor()  # No max_iter param; XGBoost uses n_estimators instead
    elif method == 'tree':
        model = DecisionTreeRegressor()
    else:
        raise ValueError(f"Unsupported method: {method}")


    model.fit(features, Y)

    def predict_fn(features_new):
        if method == 'logistic':
            return model.predict_proba(features_new)[:, 1] # type: ignore[reportAttributeAccessIssue]
        else:
            pred = model.predict(features_new)
            # if method == 'mlp':
            #     plt.figure(figsize=(6, 4))
            #     plt.plot(model.loss_curve_)  # type: ignore[reportAttributeAccessIssue]
            #     plt.title("MLPRegressor Loss Curve")
            #     plt.xlabel("Iteration")
            #     plt.ylabel("Loss")
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.show()
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred[:, 0]
            return pred

    return predict_fn



###############################
def lm_build_all_psi(X, Y, W1, W2, V, theta, method='mlp'):
    """
    Build three OLS-based ψ functions of the form:
        ψ(X, Y_like) = (E[XXᵀ])⁻¹ · X · (Y_like − Xᵀθ)

    Three versions differ by Y_like:
        ψ₁: uses observed Y
        ψ₂: uses predicted Ŷ from (X, W1, W2, V)
        ψ₃: uses predicted Ŷ from (X, W1, W2)

    Requires inputs already in 2D where appropriate:
        X  : ndarray of shape (n, d)
        Y  : ndarray of shape (n,) or (n, 1)
        W1, W2, V : ndarray of shape (n, 1)

    Returns:
        psi_1, psi_2, psi_3 functions
    """
    # Precompute inverse moment matrix
    n, d = X.shape
    XX_inv = np.linalg.inv((X.T @ X) / n)

    # Fit psi models
    features_3 = np.concatenate([X, W1, W2], axis=1)
    f3 = lm_fit_y_general(features_3, Y.reshape(-1), method=method)

    features_2 = np.concatenate([X, W1, W2, V], axis=1)
    f2 = lm_fit_y_general(features_2, Y.reshape(-1), method=method)

    def psi_1(X_val, Y_val):
        X_val = X_val.reshape(-1, X.shape[1])    # (n_val, d)
        Y_val = Y_val.reshape(-1, 1)             # (n_val, 1)
        residual = Y_val - X_val @ theta         # (n_val, 1)
        result = XX_inv @ (X_val.T @ residual)   # (d, 1)
        return result.reshape(-1)                # (d,)

    def psi_2(X_val, W1_val, W2_val, V_val):
        X_val = X_val.reshape(-1, X.shape[1])
        W1_val = W1_val.reshape(-1, 1)
        W2_val = W2_val.reshape(-1, 1)
        V_val  = V_val.reshape(-1, 1)
        feats = np.concatenate([X_val, W1_val, W2_val, V_val], axis=1)  # (n_val, d+3)
        Yhat = f2(feats).reshape(-1, 1)             # (n_val, 1)
        residual = Yhat - X_val @ theta             # (n_val, 1)
        result = XX_inv @ (X_val.T @ residual)      # (d, 1)
        return result.reshape(-1)                   # (d,)

    def psi_3(X_val, W1_val, W2_val):
        X_val = X_val.reshape(-1, X.shape[1])
        W1_val = W1_val.reshape(-1, 1)
        W2_val = W2_val.reshape(-1, 1)
        feats = np.concatenate([X_val, W1_val, W2_val], axis=1)  # (n_val, d+2)
        Yhat = f3(feats).reshape(-1, 1)             # (n_val, 1)
        residual = Yhat - X_val @ theta             # (n_val, 1)
        result = XX_inv @ (X_val.T @ residual)      # (d, 1)
        return result.reshape(-1)                   # (d,)
    return psi_1, psi_2, psi_3



###############################
def general_build_all_phi(psi_2, psi_3, alpha1, alpha2, alpha3):
    """
    Construct φ₁, φ₂, φ₃ from ψ₂, ψ₃ and weighting functions α₁, α₂, α₃.

    Definitions:
        φ₁ = - [α₂ / ((α₁ + α₂) * α₁)] * ψ₂ - [α₃ / (α₁ + α₂)] * ψ₃
        φ₂ = [1 / (α₁ + α₂)] * ψ₂ - [α₃ / (α₁ + α₂)] * ψ₃
        φ₃ = ψ₃

    Shapes:
        Inputs  : X_val (d,), W1_val (1,), W2_val (1,), V_val (1,)
        Outputs : φⱼ(X_val, W1_val, W2_val, V_val) → (d,)
    """

    def phi_1(X_val, W1_val, W2_val, V_val):
        a1 = alpha1(X_val, W1_val, W2_val)
        a2 = alpha2(X_val, W1_val, W2_val)
        a3 = alpha3(X_val, W1_val, W2_val)

        denom_sum = a1 + a2
        denom_prod = denom_sum * a1

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        term1 = - (a2 / denom_prod) * psi2_val
        term2 = - (a3 / denom_sum) * psi3_val

        return term1 + term2

    def phi_2(X_val, W1_val, W2_val, V_val):
        a1 = alpha1(X_val, W1_val, W2_val)
        a2 = alpha2(X_val, W1_val, W2_val)
        a3 = alpha3(X_val, W1_val, W2_val)

        denom_sum = a1 + a2

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        term1 = (1 / denom_sum) * psi2_val
        term2 = (a3 / denom_sum) * psi3_val

        return term1 - term2

    def phi_3(X_val, W1_val, W2_val, V_val):
        return psi_3(X_val, W1_val, W2_val)

    return phi_1, phi_2, phi_3



###############################
def general_estimate_moments_mcar(psi_1, phi_1, phi_2, phi_3, X, Y, W1, W2, V):
    """
    Estimate second moments and cross-moments under MCAR setting.

    Args:
        psi_1: Function of (X_val, Y_val) → shape (d,)
        phi_1, phi_2, phi_3: Functions of (X_val, W1_val, W2_val, V_val) → shape (d,)
        X: ndarray (n, d)
        Y: ndarray (n,)
        W1, W2, V: ndarrays (n,)

    Returns:
        A dictionary with:
            - "E[ψ₁ψ₁ᵀ]":       shape (d, d)
            - "E[φⱼφⱼᵀ]":       list of 3 matrices, each (d, d)
            - "Cov(ψ₁, φⱼ)":    list of 3 matrices, each (d, d)
    """
    n, d = X.shape
    # Stack ψ₁ values
    psi1_vals = np.stack([psi_1(X[i], Y[i]) for i in range(n)])  # shape: (n, d)
    # Stack φⱼ values
    phi_vals = []
    for phi_fn in [phi_1, phi_2, phi_3]:
        phi_i = np.stack([phi_fn(X[i], W1[i], W2[i], V[i]) for i in range(n)])  # shape: (n, d)
        phi_vals.append(phi_i)

    # E[ψ₁ ψ₁ᵀ]
    E_psi1_psi1T = psi1_vals.T @ psi1_vals / n

    # E[φⱼ φⱼᵀ] and Cov(ψ₁, φⱼ)
    E_phi_phiT_list = []
    Cov_psi_phi_list = []
    for phi_j_vals in phi_vals:
        E_phi_phiT = phi_j_vals.T @ phi_j_vals / n
        Cov_psi_phi = psi1_vals.T @ phi_j_vals / n
        E_phi_phiT_list.append(E_phi_phiT)
        Cov_psi_phi_list.append(Cov_psi_phi)

    return {
    "E[psi1 psi1^T]": E_psi1_psi1T,
    "E[phi_j phi_j^T]": E_phi_phiT_list,
    "Cov(psi1, phi_j)": Cov_psi_phi_list,
}



###############################
def general_estimate_m_matrix_mcar(moment_dict, alphas):
    """
    Estimate the matrix M^{(1)} under MCAR using moment estimators.

    Formula:
        M = - Cov(ψ₁, φ₁) × [ Σⱼ αⱼ E[φⱼ φⱼᵀ] ]⁻¹

    Args:
        moment_dict: A dictionary with keys:
            - "Cov(ψ1, φ1)"      : ndarray of shape (d, d)
            - "E[φⱼφⱼᵀ]"         : list of 3 ndarrays, each of shape (d, d)
        alpha_weights: List or array of length 3 with scalar weights αⱼ

    Returns:
        M: ndarray of shape (d, d)
    """
    cov_psi1_phi1 = moment_dict["Cov(psi1, phi_j)"][0]           # shape: (d, d)
    E_phi_phiT_list = moment_dict["E[phi_j phi_j^T]"]            # list of 3 (d, d) matrices
    # Weighted sum: Σⱼ αⱼ E[φⱼ φⱼᵀ]
    weighted_sum = sum(alpha * E_phi for alpha, E_phi in zip(alphas, E_phi_phiT_list))

    # Inverse and multiply
    M = - cov_psi1_phi1 @ np.linalg.inv(weighted_sum)
    return M



###############################
def general_estimate_variance_mcar(moment_dict: dict,
                                   alphas: np.ndarray) -> np.ndarray:
    """
    Compute  Cov̂(θ̂)  for the MCAR efficient estimator using the formula

        Cov̂(θ̂)
        =  Ê[ψ₁ ψ₁ᵀ] / α₁
          - Cov̂(ψ₁, φ̂₁) · ( Σ_{j=1}³ αⱼ Ê[φ̂ⱼ φ̂ⱼᵀ] )⁻¹ · Cov̂(ψ₁, φ̂₁)ᵀ

    Parameters
    ----------
    moment_dict : dict
        Output of `general_estimate_moments_mcar`, with keys
          "E[psi1 psi1^T]"   : (d, d) ndarray
          "E[phi_j phi_j^T]" : list of 3 (d, d) ndarrays
          "Cov(psi1, phi_j)" : list of 3 (d, d) ndarrays
    alphas : ndarray, shape-(3,)
        Vector (alpha1, alpha2, alpha3).

    Returns
    -------
    cov_theta : ndarray (d, d)
        Estimated covariance matrix of the efficient estimator.
    """
    # unpack moments
    E_psi1_psi1T = moment_dict["E[psi1 psi1^T]"]          # (d, d)
    E_phi_phiT_list = moment_dict["E[phi_j phi_j^T]"]     # list of 3 (d, d)
    Cov_psi_phi_list = moment_dict["Cov(psi1, phi_j)"]    # list of 3 (d, d)

    alpha1, alpha2, alpha3 = alphas
    # Σ αⱼ E[φ_j φ_jᵀ]
    weighted_phi_cov = (alpha1 * E_phi_phiT_list[0]
                        + alpha2 * E_phi_phiT_list[1]
                        + alpha3 * E_phi_phiT_list[2])

    inv_weighted_phi_cov = np.linalg.inv(weighted_phi_cov)
    cov_psi_phi1 = Cov_psi_phi_list[0]            # only φ̂₁ appears

    # compute covariance estimator
    correction = cov_psi_phi1 @ inv_weighted_phi_cov @ cov_psi_phi1.T
    cov_theta = E_psi1_psi1T / alpha1 - correction
    return cov_theta



###############################
def lm_mono_debias_estimate_mcar(
        X, Y, W1, W2, V, R,
        phi_list,         # [phi_1, phi_2, phi_3]
        M_hat,            # (d, d)
        w_init=None       # optional sample weights for WLS
    ):
    """
    Efficient estimator θ̂^(1) under MCAR (mono-debiasing).

    Args
    ----
    X   : (n, d) ndarray
    Y   : (n,)   ndarray
    W1, W2, V : (n,) ndarrays
    R   : (n,) integer array with values {1,2,3}  (missingness pattern)
    phi_list : list [φ₁, φ₂, φ₃]  — each (X_val, W1_val, W2_val, V_val) → (d,)
    M_hat    : (d, d) ndarray from Step 3
    w_init   : optional sample-weight vector (n,) for initial WLS

    Returns
    -------
    theta_hat : (d,) ndarray — final debiased estimator
    """
    n, d = X.shape

    # ── Step 1 : initial estimator  θ̃ via WLS/OLS ──
    mask_obs = ~np.isnan(Y).reshape(-1)              # only rows with observed Y
    #print(Y.shape)
    #print(mask_obs.shape)
    X_obs = X[mask_obs]
    Y_obs = Y[mask_obs]

    w_obs = w_init[mask_obs] if w_init is not None else None
    theta_tilde = lm_fit_wls(X_obs, Y_obs, w=w_obs)    # shape (d,)

    # ── Step 2 : correction term  (1/n) Σ_i φ_{r_i} ──
    correction = np.zeros(d)
    for i in range(n):
        r = R[i]
        if   r == 1: phi_vec = phi_list[0](X[i], W1[i], W2[i], V[i])
        elif r == 2: phi_vec = phi_list[1](X[i], W1[i], W2[i], V[i])
        elif r == 3: phi_vec = phi_list[2](X[i], W1[i], W2[i], V[i])
        else:
            raise ValueError(f"R[{i}] = {r} is not in {{1,2,3}}")
        correction += phi_vec
    correction /= n                      # (d,)

    # ── Step 3 : final debiased estimator ──
    theta_hat = theta_tilde + M_hat @ correction
    return theta_hat



###############################
def lm_mono_debias_estimate_mcar_crossfit(
        X, Y, W1, W2, V, R,
        alpha,                   # tuple/list (α1, α2, α3) – required
        method: str = "mlp"):
    """
    End-to-end 3-fold cross-fit pipeline (Steps 1-6).

    Parameters
    ----------
    X : ndarray (n, d)
        Feature matrix.
    Y : ndarray (n,)
        Response (may contain NaN).
    W1, W2, V : ndarray (n,)
        Auxiliary model outputs / variables.
    R : ndarray (n,)
        Missingness pattern, values in {1, 2, 3}.
    alpha : (α1, α2, α3)
        MCAR probabilities.
    method : str
        Base learner for lm_build_all_psi (default “mlp”).

    Returns
    -------
    theta_final : ndarray (d,)
        Cross-fitted efficient estimator.
    """

    # ---------- Step 1 : deterministic 3-way split ----------
    n, d = X.shape
    idx1, idx2, idx3 = sample_split_three(n)
    folds = [
        (idx1, idx2, idx3),  # rotation 1  (D1, D2, D3)
        (idx2, idx3, idx1),  # rotation 2
        (idx3, idx1, idx2)   # rotation 3
    ]

    # constant-alpha functions (MCAR)
    alpha1_fn = lambda Xv, w1v, w2v: np.full(len(w1v), alpha[0])
    alpha2_fn = lambda Xv, w1v, w2v: np.full(len(w1v), alpha[1])
    alpha3_fn = lambda Xv, w1v, w2v: np.full(len(w1v), alpha[2])
    alpha_vec = np.array(alpha)

    def subset(idxs):
        """Return (X, Y, W1, W2, V, R) restricted to idxs."""
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]

    theta_list, cov_list, size_list = [], [], []

    # ---------- 3-fold rotation loop ----------
    for D1, D2, D3 in folds:

        # fetch data subsets
        X1, Y1, W11, W21, V1, R1 = subset(D1)
        X2, Y2, W12, W22, V2, R2 = subset(D2)
        X3, Y3, W13, W23, V3, R3 = subset(D3)

        # ----- Step 2 : build ψ and φ on D3 ∩ {R = 1} -----
        mask_r1_D3 = (R3 == 1)
        theta_pre = lm_fit_wls(
            X3[mask_r1_D3],
            Y3[mask_r1_D3])

        psi_1, psi_2, psi_3 = lm_build_all_psi(
            X3[mask_r1_D3], Y3[mask_r1_D3],
            W13[mask_r1_D3], W23[mask_r1_D3], V3[mask_r1_D3],
            theta_pre, method)

        phi_1, phi_2, phi_3 = general_build_all_phi(
            psi_2, psi_3,
            alpha1_fn, alpha2_fn, alpha3_fn)

        # ----- Step 3 : estimate covariance matrices on D2 ∩ {R = 1} -----
        mask_r1_D2 = (R2 == 1)
        moments = general_estimate_moments_mcar(
            psi_1, phi_1, phi_2, phi_3,
            X2[mask_r1_D2], Y2[mask_r1_D2],
            W12[mask_r1_D2], W22[mask_r1_D2], V2[mask_r1_D2])

        # Estimate M matrix
        M_hat = general_estimate_m_matrix_mcar(moments, alpha_vec)

        # ----- Step 4 : debias on D1 -----
        theta_k = lm_mono_debias_estimate_mcar(
            X1, Y1, W11, W21, V1, R1,
            [phi_1, phi_2, phi_3], M_hat)
        cov_k = general_estimate_variance_mcar(moments, alpha_vec)

        theta_list.append(theta_k)
        cov_list.append(cov_k)
        size_list.append(len(D1))

    # ---------- Step 6 : weighted average ----------
    weights = np.array(size_list) / n
    theta_final = sum(w * th for w, th in zip(weights, theta_list))
    cov_final   = sum(w * cv for w, cv in zip(weights, cov_list))
    return theta_final, cov_final



def general_build_all_phi_function(psi_2, psi_3):
    """
    Return three functions phi_1, phi_2, phi_3, each explicit in `alpha`.
    
    Each returned function has signature
        phi_j(alpha, X_val, W1_val, W2_val, V_val) -> (d,)
    
    Parameters
    ----------
    psi_2 : callable
        psi_2(X, W1, W2, V) -> (d,)
    psi_3 : callable
        psi_3(X, W1, W2) -> (d,)
    
    Returns
    -------
    phi_1, phi_2, phi_3 : callables
        Each accepts:
            alpha : array-like shape (3,)  # (alpha1, alpha2, alpha3)
            X_val, W1_val, W2_val, V_val  # single data point
        and returns a (d,) vector.
    """

    def phi_1(alpha, X_val, W1_val, W2_val, V_val):
        alpha1, alpha2, alpha3 = alpha
        denom_sum  = alpha1 + alpha2
        denom_prod = denom_sum * alpha1

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return -(alpha2 / denom_prod) * psi2_val - (alpha3 / denom_sum) * psi3_val

    def phi_2(alpha, X_val, W1_val, W2_val, V_val):
        alpha1, alpha2, alpha3 = alpha
        denom_sum = alpha1 + alpha2

        psi2_val = psi_2(X_val, W1_val, W2_val, V_val)
        psi3_val = psi_3(X_val, W1_val, W2_val)

        return (1 / denom_sum) * psi2_val - (alpha3 / denom_sum) * psi3_val

    def phi_3(alpha, X_val, W1_val, W2_val, V_val):
        # phi_3 = psi_3  (independent of alpha)
        return psi_3(X_val, W1_val, W2_val)

    return phi_1, phi_2, phi_3



def general_estimate_moments_function_mcar(
        psi_1,
        phi_1, phi_2, phi_3,
        X, Y, W1, W2, V):
    """
    Build a callable moment_fn so that users can obtain empirical
    moment matrices for any alpha without re-processing the data.

    Parameters
    ----------
    psi_1 : callable
        psi_1(X_val, Y_val) -> (d,)
    phi_1, phi_2, phi_3 : callables
        Each has signature
            phi_j(alpha, X_val, W1_val, W2_val, V_val) -> (d,)
    X : ndarray, shape (n, d)
    Y : ndarray, shape (n,)
    W1, W2, V : ndarrays, shape (n,)

    Returns
    -------
    moment_fn : callable
        moment_fn(alpha) returns a dict with
            "E[psi1 psi1^T]"   : (d, d)
            "E[phi_j phi_j^T]" : list of 3 (d, d) arrays
            "Cov(psi1, phi_j)" : list of 3 (d, d) arrays
    """
    n, d = X.shape

    # Pre-compute ψ₁ values (independent of alpha)
    psi1_vals = np.stack([psi_1(X[i], Y[i]) for i in range(n)])   # (n, d)
    E_psi1_psi1T = psi1_vals.T @ psi1_vals / n                    # (d, d)

    def moment_fn(alpha):
        """
        Compute empirical moments for a given alpha = (alpha1, alpha2, alpha3).
        """
        # Stack φ_j(alpha, ·) for all samples
        phi_vals = [
            np.stack([
                phi_func(alpha, X[i], W1[i], W2[i], V[i]) for i in range(n)
            ])
            for phi_func in (phi_1, phi_2, phi_3)
        ]

        # E[φ_j φ_j^T] and Cov(ψ₁, φ_j)
        E_phi_phiT_list = [phi_j.T @ phi_j / n for phi_j in phi_vals]
        Cov_psi_phi_list = [psi1_vals.T @ phi_j / n for phi_j in phi_vals]

        return {
            "E[psi1 psi1^T]":  E_psi1_psi1T,
            "E[phi_j phi_j^T]": E_phi_phiT_list,
            "Cov(psi1, phi_j)": Cov_psi_phi_list,
        }

    return moment_fn




def general_get_trace_variance_function_alpha_mcar(
        moment_fn,              # callable: moment_fn(alpha_vec) -> moment_dict
        tau, c,
        return_full: bool = False):
    """
    Return a function g(alpha1) that produces either
        • Tr(Cov_hat(theta))   (default) or
        • Cov_hat(theta)       (if return_full=True)
    under the budget constraint (2.3):

        alpha2 = tau - c * alpha1
        alpha3 = 1 + (c - 1) * alpha1 - tau

    Parameters
    ----------
    moment_fn : callable
        Output of `general_estimate_moments_function`.
        It maps a complete alpha vector (len-3 array) to the required moment dict.
    tau : float
        Constraint: alpha1 + alpha2 = tau
    c : float
        Constraint parameter in (2.3)
    return_full : bool, default False
        If True, g(alpha1) returns the full covariance matrix.
        Otherwise it returns its trace.

    Returns
    -------
    g : callable
        g(alpha1) -> scalar (trace) or (d, d) matrix, depending on `return_full`.
    """

    def g(alpha1: float):
        # Compose the full alpha vector under constraint (2.3)
        alpha2 = tau - c * alpha1
        alpha3 = 1.0 + (c - 1.0) * alpha1 - tau
        alpha_vec = np.array([alpha1, alpha2, alpha3])

        # Retrieve empirical moments for this alpha
        moment_dict = moment_fn(alpha_vec)
        E_psi1_psi1T   = moment_dict["E[psi1 psi1^T]"]      # (d, d)
        E_phi_phiT_lst = moment_dict["E[phi_j phi_j^T]"]    # list of (d, d)
        Cov_psi_phi1   = moment_dict["Cov(psi1, phi_j)"][0] # (d, d)

        # Sandwich covariance estimator
        weighted_phi_cov = sum(a * E for a, E in zip(alpha_vec, E_phi_phiT_lst))
        inv_weighted     = np.linalg.inv(weighted_phi_cov)

        cov_theta = E_psi1_psi1T / alpha1 - Cov_psi_phi1 @ inv_weighted @ Cov_psi_phi1.T
        return cov_theta if return_full else np.trace(cov_theta)

    return g




def lm_mono_debias_budget_constrained_obtain_alpha_mcar(
        X, Y, W1, W2, V, R,
        tau: float,
        c: float,
        method: str = "mlp",
        eps: float = 1e-6):
    """
    Two-fold cross-fitting implementation of Algorithm 2 (MCAR).
    Returns optimal alpha = (alpha1, alpha2, alpha3) that minimises
    Tr(Cov̂(θ̂)) under constraint (2.3).

    Parameters
    ----------
    X : ndarray (n, d)
    Y : ndarray (n,)
    W1, W2, V : ndarray (n,)
    R : ndarray (n,), values in {1,2,3}
    tau, c : floats   (constraint parameters)
    method : str      (base learner for ψ construction)
    eps : float       (safety margin for alpha bounds)

    Returns
    -------
    alpha_opt : ndarray (3,)
    trace_opt : float
    cov_opt   : ndarray (d, d)
    """

    # -------------------------------------------------- helpers ----------
    def sample_split_two(n_: int):
        """Random 50/50 split."""
        idx = np.random.permutation(n_)
        return idx[: n_ // 2], idx[n_ // 2 :]

    def subset(idxs):
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]

    # -------------------------------------------------- build folds ------
    n = X.shape[0]
    idx1, idx2 = sample_split_two(n)
    folds = [(idx1, idx2), (idx2, idx1)]  # two-fold cross-fit

    trace_funcs, cov_funcs = [], []

    # ====================================================== main loop ====
    for D_moment, D_build in folds:
        # ---- fetch once -----------------------------------------------
        X1, Y1, W11, W21, V1, R1 = subset(D_moment)   # moment set
        X2, Y2, W12, W22, V2, R2 = subset(D_build)    # ψ/φ   set

        mask2 = (R2 == 1)
        theta_pre = lm_fit_wls(X2[mask2], Y2[mask2])

        psi_1, psi_2, psi_3 = lm_build_all_psi(
            X2[mask2], Y2[mask2],
            W12[mask2], W22[mask2], V2[mask2],
            theta_pre, method
        )

        phi_1, phi_2, phi_3 = general_build_all_phi_function(psi_2, psi_3)

        mask1 = (R1 == 1)
        moment_fn = general_estimate_moments_function_mcar(
            psi_1, phi_1, phi_2, phi_3,
            X1[mask1], Y1[mask1],
            W11[mask1], W21[mask1], V1[mask1]
        )

        trace_f = general_get_trace_variance_function_alpha_mcar(
            moment_fn, tau, c, return_full=False)
        cov_f   = general_get_trace_variance_function_alpha_mcar(
            moment_fn, tau, c, return_full=True)

        trace_funcs.append(trace_f)
        cov_funcs.append(cov_f)

    # -------------------------------------------------- feasible range ---
    upper1 = tau / c                          # α2 ≥ 0
    lower1 = (tau - 1.0) / (c - 1.0) if c != 1.0 else 0.0  # α3 ≥ 0

    alpha1_min = max(eps, lower1)
    alpha1_max = min(1.0 - eps, upper1)
    if alpha1_min >= alpha1_max:
        raise ValueError("No feasible alpha1 interval for given τ and c")

    # -------------------------------------------------- objective -------
    def objective(alpha1: float) -> float:
        # Here alpha1 is already restricted to [alpha1_min, alpha1_max]
        return float(np.mean([f(alpha1) for f in trace_funcs]))

    res: OptimizeResult = minimize_scalar(
        objective,
        bounds=(alpha1_min, alpha1_max),
        method="bounded"
    )
    alpha1_opt = float(res.x)
    alpha2_opt = tau - c * alpha1_opt
    alpha3_opt = 1.0 + (c - 1.0) * alpha1_opt - tau
    alpha_opt  = np.array([alpha1_opt, alpha2_opt, alpha3_opt])

    # -------------------------------------------------- final cov & trace
    cov_opt   = np.mean([f(alpha1_opt) for f in cov_funcs], axis=0)
    trace_opt = float(np.trace(cov_opt))

    return alpha_opt, trace_opt, cov_opt




def lm_mono_debias_budget_constrained_obtain_alpha_mcar_var1(
        X, Y, W1, W2, V, R,
        tau: float,
        c: float,
        method: str = "mlp",
        eps: float = 1e-6):
    """
    Two-fold cross-fitting version of Algorithm 2 (MCAR) that chooses
    alpha = (α1, α2, α3) to minimise Var(θ̂_1) subject to the same
    budget constraint (2.3).

    Parameters
    ----------
    X : ndarray (n, d)
    Y : ndarray (n, 1)
    W1, W2, V : ndarray (n, 1)
    R : ndarray (n, 1)  taking values in {1,2,3}
    tau, c : floats     (constraint parameters)
    method : str        (base learner used inside ψ-construction)
    eps : float         (safety margin for alpha bounds)

    Returns
    -------
    alpha_opt : ndarray (3,)      – optimal (α1, α2, α3)
    var1_opt  : float             – optimal Var(θ̂_1)
    cov_opt   : ndarray (d, d)    – full covariance matrix at α_opt
    """

    # ------------- helpers ------------------------------------------------
    def sample_split_two(n_):
        """Return a random 50-50 split of indices."""
        idx = np.random.permutation(n_)
        return idx[: n_ // 2], idx[n_ // 2:]

    def subset(idxs):
        return X[idxs], Y[idxs], W1[idxs], W2[idxs], V[idxs], R[idxs]

    # ------------- build folds -------------------------------------------
    n = X.shape[0]
    idx1, idx2 = sample_split_two(n)
    folds = [(idx1, idx2), (idx2, idx1)]          # two-fold cross-fit

    var1_funcs, cov_funcs = [], []

    for D_moment, D_build in folds:
        # Moment (fold-1) and building (fold-2) sets
        X1, Y1, W11, W21, V1, R1 = subset(D_moment)
        X2, Y2, W12, W22, V2, R2 = subset(D_build)

        mask2  = (R2 == 1)
        theta_pre = lm_fit_wls(X2[mask2], Y2[mask2])

        psi_1, psi_2, psi_3 = lm_build_all_psi(
            X2[mask2], Y2[mask2],
            W12[mask2], W22[mask2], V2[mask2],
            theta_pre, method
        )

        phi_1, phi_2, phi_3 = general_build_all_phi_function(psi_2, psi_3)

        mask1 = (R1 == 1)
        moment_fn = general_estimate_moments_function_mcar(
            psi_1, phi_1, phi_2, phi_3,
            X1[mask1], Y1[mask1],
            W11[mask1], W21[mask1], V1[mask1]
        )

        # Full covariance function
        cov_f = general_get_trace_variance_function_alpha_mcar(
            moment_fn, tau, c, return_full=True
        )

        # Extract only the (0,0) entry (Var θ̂_1)
        var1_f = lambda a1, f=cov_f: float(f(a1)[0, 0])

        var1_funcs.append(var1_f)
        cov_funcs.append(cov_f)

    # ------------- feasible range for α1 ---------------------------------
    upper1 = tau / c
    lower1 = (tau - 1.0) / (c - 1.0) if c != 1.0 else 0.0

    alpha1_min = max(eps, lower1)
    alpha1_max = min(1.0 - eps, upper1)
    if alpha1_min >= alpha1_max:
        raise ValueError("No feasible α1 interval for given τ and c")

    # ------------- objective (mean Var(θ̂₁)) -----------------------------
    def objective(alpha1: float) -> float:
        return float(np.mean([f(alpha1) for f in var1_funcs]))

    res = minimize_scalar(objective,
                          bounds=(alpha1_min, alpha1_max),
                          method="bounded")
    alpha1_opt = float(res.x)  # type: ignore[arg-type]
    alpha2_opt = tau - c * alpha1_opt
    alpha3_opt = 1.0 + (c - 1.0) * alpha1_opt - tau
    alpha_opt  = np.array([alpha1_opt, alpha2_opt, alpha3_opt])

    # ------------- final covariance and Var(θ̂₁) ------------------------
    cov_opt  = np.mean([f(alpha1_opt) for f in cov_funcs], axis=0)
    var1_opt = float(cov_opt[0, 0])

    return alpha_opt, var1_opt, cov_opt
