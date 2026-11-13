"""Position-wise feed-forward networks."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


def _relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu2": _relu2,
}


class SwiGLU(nn.Module):
    r"""Gated feed-forward layer.

    .. math::  \operatorname{FFN}(x) = W_2\big(\operatorname{SiLU}(W_1 x) \odot W_3 x\big)

    ``W_1`` produces the gate, ``W_3`` the value; their product is projected
    back by ``W_2``. No biases.
    """

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0, activation: str = "silu") -> None:
        super().__init__()
        self.act = ACTIVATIONS[activation]
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(self.act(self.w1(x)) * self.w3(x)))


class GELUMLP(nn.Module):
    r"""Classic Transformer MLP: :math:`W_2\,\operatorname{GELU}(W_1 x)`."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0, activation: str = "gelu") -> None:
        super().__init__()
        self.act = ACTIVATIONS[activation]
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


def build_feedforward(
    dim: int, hidden: int, use_swiglu: bool = True, dropout: float = 0.0, activation: str | None = None
) -> nn.Module:
    if use_swiglu:
        return SwiGLU(dim, hidden, dropout, activation or "silu")
    return GELUMLP(dim, hidden, dropout, activation or "gelu")
