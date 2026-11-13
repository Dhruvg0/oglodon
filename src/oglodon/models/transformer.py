"""A single pre-norm Transformer block."""

from __future__ import annotations

import torch
from torch import nn

from ..config import OglodonConfig
from ..layers.attention import CausalSelfAttention
from ..layers.cache import KVCache
from ..layers.feedforward import build_feedforward
from ..layers.normalization import build_norm


class TransformerBlock(nn.Module):
    r"""Pre-norm residual block.

    .. math::
        h = x + \operatorname{Attn}(\operatorname{Norm}_1(x)), \qquad
        y = h + \operatorname{FFN}(\operatorname{Norm}_2(h))
    """

    def __init__(self, config: OglodonConfig, layer_idx: int) -> None:
        super().__init__()
        c = config
        self.attn_norm = build_norm(c.hidden_size, c.use_rmsnorm, c.norm_eps)
        self.attn = CausalSelfAttention(
            c.hidden_size,
            c.num_heads,
            c.num_kv_heads,
            c.dropout,
            layer_idx,
            c.attn_impl,
            c.sliding_window,
            c.qkv_bias,
            c.use_alibi,
        )
        self.ffn_norm = build_norm(c.hidden_size, c.use_rmsnorm, c.norm_eps)
        self.ffn = build_feedforward(c.hidden_size, c.intermediate_size, c.use_swiglu, c.dropout, c.ffn_act)
        self.drop = nn.Dropout(c.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_padding_mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        a, weights = self.attn(self.attn_norm(x), rope, key_padding_mask, cache, output_attentions)
        x = x + self.drop(a)
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x, weights
