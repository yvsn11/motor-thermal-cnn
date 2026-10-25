"""Models: causal dilated residual CNN, GRU and a ridge baseline."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch import nn

from .data import ridge_features

N_IN, N_OUT = 8, 4


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CausalBlock(nn.Module):
    """Dilated causal conv (left padding only) -> GELU -> 1x1 conv, with a residual connection."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.pad = dilation * (kernel - 1)
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.pw = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.pw.weight)  # blocks start as identity -> stable deep stack
        nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        h = self.conv(F.pad(x, (self.pad, 0)))
        return x + self.pw(self.drop(F.gelu(h)))


class CausalCNN(nn.Module):
    """TCN-style causal dilated residual CNN. Input (B, T, 8), output (B, T, 4)."""

    def __init__(
        self,
        channels: int = 32,
        n_blocks: int = 9,
        kernel: int = 3,
        dropout: float = 0.0,
        n_in=N_IN,
        n_out=N_OUT,
    ):
        super().__init__()
        self.kernel = kernel
        self.dilations = [2**i for i in range(n_blocks)]
        self.inp = nn.Conv1d(n_in, channels, 1)
        self.blocks = nn.ModuleList(
            CausalBlock(channels, kernel, d, dropout) for d in self.dilations
        )
        self.head = nn.Conv1d(channels, n_out, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(d * (self.kernel - 1) for d in self.dilations)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return self.head(F.gelu(h)).transpose(1, 2)


class GRUNet(nn.Module):
    """Stacked GRU followed by a linear head. Input (B, T, 8), output (B, T, 4)."""

    def __init__(self, hidden: int = 64, layers: int = 2, n_in=N_IN, n_out=N_OUT):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h)


def build_model(kind: str, **kw) -> nn.Module:
    if kind == "cnn":
        return CausalCNN(**kw)
    if kind == "gru":
        return GRUNet(**kw)
    raise ValueError(kind)


class RidgeBaseline:
    """Ridge regression on causal EWMA features (inputs only, no target lags)."""

    def __init__(self, alpha: float = 1.0, spans=(10, 60, 300, 1200, 4800)):
        self.alpha = alpha
        self.spans = tuple(spans)
        self.model = Ridge(alpha=alpha)

    def _feats(self, x):
        f = ridge_features(x, self.spans)
        return (f - self.mu) / self.sd

    def fit(self, profiles_x: list, profiles_y: list):
        raw = np.concatenate([ridge_features(x, self.spans) for x in profiles_x])
        self.mu = raw.mean(0)
        self.sd = np.maximum(raw.std(0), 1e-8)
        self.model.fit((raw - self.mu) / self.sd, np.concatenate(profiles_y))
        return self

    def predict(self, x):
        """x: one profile (T, 8) of raw inputs -> (T, 4) predictions in kelvin."""
        return self.model.predict(self._feats(x)).astype(np.float32)

    @property
    def n_params(self) -> int:
        return int(self.model.coef_.size + self.model.intercept_.size)


def set_threads(n: int = 2) -> None:
    torch.set_num_threads(n)
