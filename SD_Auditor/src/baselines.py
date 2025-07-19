import numpy as np
from src.mono_debais import lm_fit_wls
def lm_ols_estimate(X: np.ndarray,
                         Y: np.ndarray):
    """
    Ordinary Least Squares (OLS) under MCAR (Missing Completely At Random),
    **without an explicit intercept term**.

    Parameters
    ----------
    X : (n, d) ndarray
        Design matrix.
    Y : (n,) or (n, 1) ndarray
        Response vector; NaNs indicate missing outcomes.

    Returns
    -------
    theta_hat : (d,) ndarray
        OLS estimator obtained by minimizing
            ‖Y_obs − X_obs θ‖₂²
        over the subset of rows where Y is observed.
    """

    # Step 1 – keep only rows with observed Y
    mask_obs = ~np.isnan(Y).reshape(-1)
    X_obs    = X[mask_obs]
    Y_obs    = Y[mask_obs]

    # Step 2 – call existing weighted least-squares routine; 
    #           sample_weight=None → ordinary (unweighted) LS
    theta_hat = lm_fit_wls(X_obs, Y_obs, w=None)

    return theta_hat