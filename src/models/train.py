from typing import Optional, Literal

import torch
from torch.utils.data import DataLoader, TensorDataset

from .builders import get_model, get_loss
from ..utils import get_device            # relative import from src.utils

__all__ = ["make_loader", "train_model"]  # exported symbols
# --------------------------------------------------------------------- #
# Helper: pick a scheduler object given its name
# --------------------------------------------------------------------- #
def _make_scheduler(optimizer, name: str, epochs: int, kw: Optional[dict]):
    kw = {} if kw is None else kw
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, **kw
        )
    elif name == "step":
        # e.g. kw={"step_size":100, "gamma":0.5}
        return torch.optim.lr_scheduler.StepLR(optimizer, **kw)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler '{name}'")

# ----------------------------------------------------------------------
def make_loader(X, y, idx, *, batch_size=50000, shuffle=True):
    """
    Return a DataLoader backed by *views* (no extra copy) of X[idx], y[idx].
    Assumes X, y are already on the correct device.
    """
    ds = TensorDataset(X[idx], y[idx])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    X, y,
    *,
    method: Literal["mlp", "mlpclass", "linreg", "logistic"] = "mlp",
    epochs: int = 250,
    batch_size: int = 50000,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    # ---- NEW options ----
    scheduler_name: str = "cosine",      # {"cosine", "step", "none"}
    scheduler_kw: Optional[dict] = None, # extra kwargs for the scheduler
    log_interval: int = 50,              # how often to print loss / LR
):
    """
    Fit *method* on (X, y) and return a trained ``nn.Module``.

    Parameters
    ----------
    scheduler_name : str
        Type of LR scheduler: "cosine" (default), "step", or "none".
    scheduler_kw : dict or None
        Extra arguments passed to the scheduler constructor.
        • For "step":  {"step_size": 100, "gamma": 0.5}, etc.
    log_interval : int
        Print average loss & current LR every *log_interval* epochs.
    """
    device = get_device() if device is None else device
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1, 1)

    model = get_model(method, X.shape[1]).to(device)
    loss_fn, _ = get_loss(method)

    # Closed-form solution for linear regression
    if method == "linreg":
        with torch.no_grad():
            w, *_ = torch.linalg.lstsq(X, y)
        model.w.weight.data.copy_(w.T)
        model.w.bias.data.zero_()
        return model.eval()

    # -------- SGD loop -------- #
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _make_scheduler(optim, scheduler_name, epochs, scheduler_kw)
    loader = make_loader(X, y, slice(None), batch_size=batch_size)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, batch_count = 0.0, 0

        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            batch_count += 1

        if scheduler is not None:
            scheduler.step()

        # ---- logging ----
        # if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
        #     avg_loss = total_loss / max(batch_count, 1)
        #     current_lr = optim.param_groups[0]["lr"]
        #     print(f"Epoch {epoch:3d}/{epochs} | lr={current_lr:>.2e} | "
        #           f"avg_loss={avg_loss:.6f}")

    return model.eval()