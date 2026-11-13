"""Rotary position embeddings (RoPE)."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    r"""Pre-computes the rotation angles used by RoPE.

    Each pair of channels ``(x_{2i}, x_{2i+1})`` of a query/key vector is
    rotated by the angle :math:`m\,\theta_i` where ``m`` is the token position
    and :math:`\theta_i = \text{base}^{-2i/d}`. Because rotations compose, the
    dot product ``q_m . k_n`` depends only on the *relative* offset ``m - n``.

    The tables are non-persistent buffers: they are re-derived from the config
    and never written to checkpoints.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        # Tables are built in float64 so a float64 model keeps full precision; float32/bf16 models
        # round them when moved (`.to(dtype)`), exactly as before.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        angles = torch.outer(torch.arange(max_seq_len, dtype=torch.float64), inv_freq)  # (T, d/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up ``cos``/``sin`` for ``position_ids`` of shape ``(B, T)``.

        Returns tensors of shape ``(B, 1, T, head_dim/2)`` (broadcast over heads).
        """
        cos, sin = cast(torch.Tensor, self.cos), cast(torch.Tensor, self.sin)  # registered buffers
        return cos[position_ids].unsqueeze(1), sin[position_ids].unsqueeze(1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs of ``x`` with shape ``(B, H, T, head_dim)``.

    Uses the interleaved pairing ``(0,1), (2,3), ...`` as in the original
    MiniGPT notebook.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    # rotate in at least float32 (float64 stays float64) and cast back: no-op for float32 models
    ct = torch.promote_types(x.dtype, torch.float32)
    x1, x2 = x1.to(ct), x2.to(ct)
    cos, sin = cos.to(ct), sin.to(ct)
    rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)
