# ── standard library ───────────────────────────────────────────────────
from typing import Optional, Sequence, Tuple

# ── third-party libraries ──────────────────────────────────────────────
import torch
from torch.distributions import MultivariateNormal

# ── intra-package utilities ────────────────────────────────────────────
from . import utils  # set_global_seed, get_device, …

# ======================================================================
# Private helpers
# ======================================================================


def _as_tensor(
    x,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cast *x* (NumPy array, list, scalar, or Tensor) to a tensor on *device*."""
    device = utils.get_device() if device is None else device
    return torch.as_tensor(x, dtype=dtype, device=device)


def _sample_mv(
    n: int,
    d: int,
    cov: Optional[torch.Tensor],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = utils.get_device() if device is None else device
    mean = torch.zeros(d, device=device)
    if cov is None:
        cov = torch.eye(d, device=device)
    else:
        cov = _as_tensor(cov, device=device)
    dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
    return dist.sample((n,))


# ======================================================================
# 1) Complete-data generator
# ======================================================================


def lm_generate_complete_data(
    n: int,
    d_x: int,
    d_u1: int,
    d_u2: int,
    theta_star,
    beta1_star,
    beta2_star,
    *,
    Sigma_X: Optional[torch.Tensor] = None,
    Sigma_U1: Optional[torch.Tensor] = None,
    Sigma_U2: Optional[torch.Tensor] = None,
    sigma_eps: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Draw complete data for the Gaussian linear-mixture model

        X  ~ 𝓝(0, Σ_X)      shape (n, d_x)
        U1 ~ 𝓝(0, Σ_U1)     shape (n, d_u1)
        U2 ~ 𝓝(0, Σ_U2)     shape (n, d_u2)

        Y  = X·θ* + U1·β1* + U2·β2* + ε
        W1 = X·θ* + U1·β1*             + ε₂
        W2 = X·θ*             + U2·β2* + ε₃
        V  = 𝟙{|W1 − Y| ≤ |W2 − Y|}

    Returns
    -------
    (X, U1, U2, Y, W1, W2, V)
        The first three tensors have shape ``(n, d_·)``,
        the last four are column vectors ``(n, 1)``.
    """
    device = utils.get_device() if device is None else device

    # Parameters → tensors on target device
    theta_star = _as_tensor(theta_star, device=device)
    beta1_star = _as_tensor(beta1_star, device=device)
    beta2_star = _as_tensor(beta2_star, device=device)

    # 1) Latent covariates
    X = _sample_mv(n, d_x, Sigma_X, device=device)
    U1 = _sample_mv(n, d_u1, Sigma_U1, device=device)
    U2 = _sample_mv(n, d_u2, Sigma_U2, device=device)

    # 2) Independent Gaussian noise terms
    eps = torch.randn(n, device=device) * sigma_eps
    eps2 = torch.randn(n, device=device) * sigma_eps
    eps3 = torch.randn(n, device=device) * (sigma_eps * 1.5)

    # 3) Core linear parts
    X_theta = X @ theta_star
    U1_beta = U1 @ beta1_star
    U2_beta = U2 @ beta2_star

    Y  = X_theta + U1_beta + U2_beta + eps
    W1 = X_theta + U1_beta + eps2
    W2 = X_theta + U1_beta + eps3

    # 4) Preference indicator
    V = (torch.abs(W1 - Y) <= torch.abs(W2 - Y)).long()

    # 5) Shape to column vectors
    return (
        X,
        U1,
        U2,
        Y.unsqueeze(1),
        W1.unsqueeze(1),
        W2.unsqueeze(1),
        V.unsqueeze(1),
    )

# ======================================================================
# 2) MCAR missingness
# ======================================================================


def general_generate_mcar(
    X: torch.Tensor,
    Y: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    V: torch.Tensor,
    *,
    alpha: Sequence[float],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Impose MCAR missingness with probabilities (α₁, α₂, α₃):

        Pattern 1 (α₁):  no values missing
        Pattern 2 (α₂):  Y missing
        Pattern 3 (α₃):  Y and V missing
    """
    device = X.device if device is None else device
    alpha = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    if not torch.isclose(alpha.sum(), torch.tensor(1.0, device=device)):
        raise ValueError("alpha must sum to 1")

    n = Y.shape[0]

    # Draw missingness pattern R ∈ {1, 2, 3}
    R = torch.multinomial(alpha, num_samples=n, replacement=True) + 1  # (n,)
    R = R.unsqueeze(1)

    # Clone observed copies (Y, V need float to hold NaNs)
    X_obs  = X.clone()
    Y_obs  = Y.clone().float()
    W1_obs = W1.clone()
    W2_obs = W2.clone()
    V_obs  = V.clone().float()

    # Apply masks
    mask2 = R == 2
    mask3 = R == 3
    Y_obs[mask2] = torch.nan
    Y_obs[mask3] = torch.nan
    V_obs[mask3] = torch.nan

    return X_obs, Y_obs, W1_obs, W2_obs, V_obs, R

# ======================================================================
# 3) Wrapper: complete data + MCAR
# ======================================================================


def lm_generate_obs_data_mcar(
    n: int,
    d_x: int,
    d_u1: int,
    d_u2: int,
    theta_star,
    beta1_star,
    beta2_star,
    *,
    alpha: Sequence[float],
    Sigma_X: Optional[torch.Tensor] = None,
    Sigma_U1: Optional[torch.Tensor] = None,
    Sigma_U2: Optional[torch.Tensor] = None,
    sigma_eps: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Convenience wrapper: draw complete data and immediately impose MCAR.
    """
    device = utils.get_device() if device is None else device

    # 1) Complete data
    X, U1, U2, Y, W1, W2, V = lm_generate_complete_data(
        n, d_x, d_u1, d_u2,
        theta_star, beta1_star, beta2_star,
        Sigma_X=Sigma_X,
        Sigma_U1=Sigma_U1,
        Sigma_U2=Sigma_U2,
        sigma_eps=sigma_eps,
        device=device,
    )

    # 2) Apply MCAR
    return general_generate_mcar(
        X, Y, W1, W2, V,
        alpha=alpha,
        device=device,
    )