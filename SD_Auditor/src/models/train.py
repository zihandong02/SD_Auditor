from typing import Optional, Literal

import torch
from torch.utils.data import DataLoader, TensorDataset

from .builders import get_model, get_loss
from ..utils import get_device            # relative import from src.utils

__all__ = ["make_loader", "train_model"]  # exported symbols

# ----------------------------------------------------------------------
def make_loader(X, y, idx, *, batch_size=1024, shuffle=True):
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
    epochs: int = 200,
    batch_size: int = 10000,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
):
    """
    Fit *method* on (X, y) and return a trained `nn.Module` on *device*.
    """
    device = get_device() if device is None else device
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1, 1)

    model = get_model(method, X.shape[1]).to(device)
    loss_fn, is_clf = get_loss(method)

    # Closed-form solution for plain linear regression
    if method == "linreg":
        with torch.no_grad():
            w, *_ = torch.linalg.lstsq(X, y)
        model.w.weight.data.copy_(w.T)
        model.w.bias.data.zero_()
        return model.eval()

    # SGD loop
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loader = make_loader(X, y, slice(None), batch_size=batch_size)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

    return model.eval()