"""Multi-head / grouped-query / multi-query causal self-attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .cache import KVCache
from .rope import apply_rope


def build_attention_bias(
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
    key_padding_mask: torch.Tensor | None = None,
    window: int | None = None,
) -> torch.Tensor:
    """Additive mask of shape ``(B or 1, 1, q_len, kv_len)``.

    Query ``i`` (absolute position ``kv_len - q_len + i``) may attend to key
    ``j`` iff ``j <= kv_len - q_len + i`` (causal, offset by the cached prefix)
    and ``key_padding_mask[b, j]`` is true. With ``window`` (local attention) it must also hold that
    ``i_abs - j < window``. Blocked entries get the most
    negative finite value instead of ``-inf``, so a row that is entirely
    blocked (a padding query) yields a uniform softmax instead of NaN.
    """
    allowed = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(kv_len - q_len)
    if window is not None:
        allowed = allowed & torch.ones_like(allowed).triu(kv_len - q_len - window + 1)
    allowed = allowed[None, None]
    if key_padding_mask is not None:
        allowed = allowed & key_padding_mask.bool()[:, None, None, :]
    bias = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return bias.masked_fill(~allowed, torch.finfo(dtype).min)


def alibi_slopes(num_heads: int) -> list[float]:
    """ALiBi head slopes (Press et al., 2022): a geometric sequence; non powers of two interleave two sequences."""

    def power_of_two(n: int) -> list[float]:
        start = 2.0 ** (-8.0 / n)
        return [start ** (i + 1) for i in range(n)]

    if num_heads & (num_heads - 1) == 0:
        return power_of_two(num_heads)
    closest = 2 ** math.floor(math.log2(num_heads))
    return power_of_two(closest) + power_of_two(2 * closest)[0::2][: num_heads - closest]


class CausalSelfAttention(nn.Module):
    r"""Scaled dot-product attention with a configurable number of K/V heads.

    .. math::  \operatorname{Attn}(Q,K,V) = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_h}} + M\Big)V

    With ``num_kv_heads < num_heads`` every K/V head serves a *group* of
    ``num_heads // num_kv_heads`` query heads (query head ``h`` uses K/V head
    ``h // group_size``). Positional information is injected from outside via
    ``rope`` (``cos``/``sin`` tensors), keeping this module position-agnostic.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        layer_idx: int = 0,
        attn_impl: str = "eager",
        sliding_window: int | None = None,
        qkv_bias: bool = False,
        alibi: bool = False,
    ) -> None:
        super().__init__()
        self.alibi = alibi
        self.attn_impl, self.sliding_window = attn_impl, sliding_window
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.group_size = num_heads // num_kv_heads
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_padding_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)
        # Expand K/V heads so that query head h sees K/V head h // group_size.
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        kv_len = k.shape[2]
        weights = None
        plain = key_padding_mask is None and self.sliding_window is None and not self.alibi
        if self.attn_impl == "sdpa" and not output_attentions and plain and (kv_len == T or T == 1):
            # No mask is needed beyond causality: let SDPA use its causal fast path (this is what lets
            # FlashAttention kernels run on supported GPUs; an explicit float mask rules them out).
            # T == 1 with a cache attends to every cached key, so no mask at all.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, is_causal=kv_len == T and T > 1
            )
            out = out.transpose(1, 2).reshape(B, T, -1)
            return self.o_proj(out), None

        bias = build_attention_bias(T, k.shape[2], x.device, q.dtype, key_padding_mask, self.sliding_window)
        if self.alibi:
            dist = (
                torch.arange(kv_len - T, kv_len, device=x.device)[:, None]
                - torch.arange(kv_len, device=x.device)[None, :]
            ).clamp(min=0)
            slopes = torch.tensor(alibi_slopes(self.num_heads), dtype=torch.float64, device=x.device)
            bias = bias + (-slopes[None, :, None, None] * dist.double()).to(q.dtype)
        if self.attn_impl == "sdpa" and not output_attentions:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=self.attn_drop.p if self.training else 0.0
            )
            weights = None
        else:
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim) + bias
            # softmax in >= float32 even for bf16/fp16 models (no-op for float32/float64)
            weights = scores.softmax(dim=-1, dtype=torch.promote_types(q.dtype, torch.float32)).to(q.dtype)
            out = self.attn_drop(weights) @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        # Attention maps are returned only on request; they are never stored.
        return self.o_proj(out), (weights.detach() if output_attentions and weights is not None else None)
