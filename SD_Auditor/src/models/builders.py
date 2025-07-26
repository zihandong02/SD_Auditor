from typing import Dict, Tuple, Callable

import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# 1) Individual model classes
# ----------------------------------------------------------------------

class MLPRegressor(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(),
            nn.Linear(32, 32),   nn.ReLU(),
            nn.Linear(32, 32),   nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):          # (batch, n_in) → (batch, 1)
        return self.net(x)


class MLPClassifier(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(),
            nn.Linear(32, 32),   nn.ReLU(),
            nn.Linear(32, 32),   nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):          # logits, not sigmoid
        return self.net(x)


class LinearReg(nn.Module):
    """Classic linear regression y = X·w + b (MSE loss)."""
    def __init__(self, n_in: int):
        super().__init__()
        self.w = nn.Linear(n_in, 1, bias=True)

    def forward(self, x):
        return self.w(x)


class LogisticReg(nn.Module):
    """Binary logistic regression (returns logits)."""
    def __init__(self, n_in: int):
        super().__init__()
        self.w = nn.Linear(n_in, 1, bias=True)

    def forward(self, x):
        return self.w(x)

# ----------------------------------------------------------------------
# 2) Registry & helpers
# ----------------------------------------------------------------------

# method → (builder_fn, loss_fn, is_classification)
_MODEL_REGISTRY: Dict[str,
    Tuple[Callable[[int], nn.Module], Callable[[], nn.Module], bool]
] = {
    "mlp":       (MLPRegressor,  nn.MSELoss,          False),
    "mlpclass":  (MLPClassifier, nn.BCEWithLogitsLoss, True),
    "linreg":    (LinearReg,     nn.MSELoss,          False),
    "logistic":  (LogisticReg,   nn.BCEWithLogitsLoss, True),
}

def get_model(name: str, n_in: int):
    """
    Build a fresh model instance according to *name*.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    builder, *_ = _MODEL_REGISTRY[name]
    return builder(n_in)

def get_loss(name: str):
    """
    Return the matching loss function constructor and a
    boolean flag *is_classification*.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    _, loss_builder, is_clf = _MODEL_REGISTRY[name]
    return loss_builder(), is_clf