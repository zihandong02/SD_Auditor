import numpy as np
import sys, os
sys.path.append(os.path.abspath("./src"))
# print(os.path.abspath(".."))
# from src.utils import *
from src.utils import *
def lm_generate_complete_data(
    n: int,
    d_x: int,
    d_u1: int,
    d_u2: int,
    theta_star: np.ndarray,
    beta1_star: np.ndarray,
    beta2_star: np.ndarray,
    *,
    Sigma_X: np.ndarray | None = None,
    Sigma_U1: np.ndarray | None = None,
    Sigma_U2: np.ndarray | None = None,
    sigma_eps: float = 1.0,
):
    """
    Gaussian linear-mixture model (matches the screenshot).

    Randomness
    ----------
        X   ~ N(0, Σ_X)       shape (n, d_x)
        U1  ~ N(0, Σ_{U1})    shape (n, d_u1)
        U2  ~ N(0, Σ_{U2})    shape (n, d_u2)
        ε   ~ N(0, σ_ε²)      shape (n,)

    Constructs
    ----------
        Y  = X·θ* + U1·β1* + U2·β2* + ε           shape (n,)
        W1 = X·θ* + U1·β1*                       shape (n,)
        W2 = X·θ* + U2·β2*                       shape (n,)
        V  = 𝟙{|W1−Y| ≤ |W2−Y|}                  shape (n,)

    Returns
    -------
        X   : (n, d_x)
        U1  : (n, d_u1)
        U2  : (n, d_u2)
        Y   : (n, 1)
        W1  : (n, 1)
        W2  : (n, 1)
        V   : (n, 1)
    """
    # 1 Draw X, U1, U2
    if Sigma_X is None:
        X = np.random.standard_normal((n, d_x))
    else:
        X = np.random.multivariate_normal(np.zeros(d_x), Sigma_X, size=n)

    if Sigma_U1 is None:
        U1 = np.random.standard_normal((n, d_u1))
    else:
        U1 = np.random.multivariate_normal(np.zeros(d_u1), Sigma_U1, size=n)

    if Sigma_U2 is None:
        U2 = np.random.standard_normal((n, d_u2))
    else:
        U2 = np.random.multivariate_normal(np.zeros(d_u2), Sigma_U2, size=n)

    # 2 Independent noise ε
    eps = np.random.normal(0.0, sigma_eps, size=n)
    eps2 = np.random.normal(0.0, sigma_eps, size=n)
    eps3 = np.random.normal(0.0, sigma_eps*1.5, size=n)

    # 3 Core quantities
    X_theta   = X @ theta_star
    U1_beta   = U1 @ beta1_star
    U2_beta   = U2 @ beta2_star

    Y  = X_theta + U1_beta + U2_beta + eps
    W1 = X_theta + U1_beta + eps2
    W2 = X_theta + U2_beta + eps3

    # 4 Preference label
    V = (np.abs(W1 - Y) <= np.abs(W2 - Y)).astype(int)

    # 5 Return column-vector shapes for outputs
    return (
        X,
        U1,
        U2,
        Y.reshape(-1, 1),
        W1.reshape(-1, 1),
        W2.reshape(-1, 1),
        V.reshape(-1, 1),
    )


def general_generate_mcar(
    X,
    Y,
    W1,
    W2,
    V,
    *,
    alpha,         # (α1, α2, α3) — mandatory, must sum to 1
):
    """
    Impose MCAR missingness on (X, Y, W1, W2, V).

    Missingness patterns
    --------------------
    Pattern 1 (prob α₁):  no missing values  
    Pattern 2 (prob α₂):  **Y** is missing  
    Pattern 3 (prob α₃):  **Y** and **V** are missing  

    Args
    ----
    X : ndarray of shape (n, d)
        Feature matrix.
    Y : ndarray of shape (n, 1)
        Response vector as a column.
    W1: ndarray of shape (n, 1)
        First auxiliary model output (column vector).
    W2: ndarray of shape (n, 1)
        Second auxiliary model output (column vector).
    V : ndarray of shape (n, 1)
        Preference indicator in {0,1} (column vector).
    alpha : ndarray or list/tuple of length 3
        MCAR probabilities (α₁, α₂, α₃); must sum to 1.
        *Randomness is driven by the global seed ``env.SEED``.*

    Returns
    -------
    X_obs, Y_obs, W1_obs, W2_obs, V_obs : ndarrays
        Observed versions of the variables, with `np.nan` where values
        are missing.
    R : ndarray (n,)
        Missingness pattern vector taking values 1, 2, or 3.
    """
    n = len(Y)
    alpha_1, alpha_2, alpha_3 = alpha

    # Draw pattern R ∈ {1,2,3}
    R = np.random.choice([1, 2, 3], size=n, p=[alpha_1, alpha_2, alpha_3])

    # Create observed copies
    X_obs  = X.copy()
    Y_obs  = Y.astype(float).copy()
    W1_obs = W1.copy()
    W2_obs = W2.copy()
    V_obs  = V.astype(float).copy()

    # Apply missingness
    Y_obs[R == 2] = np.nan                  # pattern-2: Y missing
    Y_obs[R == 3] = np.nan                  # pattern-3: Y & V missing
    V_obs[R == 3] = np.nan

    return X_obs, Y_obs, W1_obs, W2_obs, V_obs, R


def lm_generate_obs_data_mcar(
    n: int,
    d_x: int,
    d_u1: int,
    d_u2: int,
    theta_star: np.ndarray,
    beta1_star: np.ndarray,
    beta2_star: np.ndarray,
    *,
    alpha,                          # (α1, α2, α3) — required
    Sigma_X:  np.ndarray | None = None,
    Sigma_U1: np.ndarray | None = None,
    Sigma_U2: np.ndarray | None = None,
    sigma_eps: float = 1.0,
):
    """
    Generate observed data under MCAR for the Gaussian linear-mixture model.

    Steps
    -----
    1. Call ``lm_generate_complete_data`` to obtain full data.
    2. Apply MCAR mechanism via ``general_generate_mcar``.
    3. Return only the observed quantities and the missingness mask.

    Returns
    -------
    X_obs, Y_obs, W1_obs, W2_obs, V_obs, R
        Shapes: X_obs (n, d_x), Y_obs/W1_obs/W2_obs/V_obs (n,1), R (n, q)
        where q is the total number of variables subject to missingness.
    """
    # ---------- Step 1: simulate complete data ----------
    X, U1, U2, Y, W1, W2, V = lm_generate_complete_data(
        n,
        d_x, d_u1, d_u2,
        theta_star, beta1_star, beta2_star,
        Sigma_X=Sigma_X,
        Sigma_U1=Sigma_U1,
        Sigma_U2=Sigma_U2,
        sigma_eps=sigma_eps
    )

    # ---------- Step 2: impose MCAR missingness ----------
    X_obs, Y_obs, W1_obs, W2_obs, V_obs, R = general_generate_mcar(
        X, Y, W1, W2, V,
        alpha=np.asarray(alpha)
    )

    # ---------- Step 3: return ----------
    return X_obs, Y_obs, W1_obs, W2_obs, V_obs, R