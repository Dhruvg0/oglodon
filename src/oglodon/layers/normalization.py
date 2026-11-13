"""Normalisation layers."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    r"""Root-mean-square normalisation.

    .. math::  y = \frac{x}{\sqrt{\operatorname{mean}(x^2) + \varepsilon}} \odot g

    Unlike LayerNorm there is no mean subtraction and no bias, which makes it
    slightly cheaper while training just as stably in practice.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute statistics in float32 for stability under half precision.
        xf = x.to(torch.promote_types(x.dtype, torch.float32))
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight


def build_norm(dim: int, use_rmsnorm: bool = True, eps: float = 1e-6) -> nn.Module:
    """Return ``RMSNorm`` or ``nn.LayerNorm`` according to the config flag."""
    return RMSNorm(dim, eps) if use_rmsnorm else nn.LayerNorm(dim, eps=eps)
