"""Key/value cache for incremental decoding."""

from __future__ import annotations

import torch


class KVCache:
    """Per-layer cache of already-computed keys and values.

    Keys/values have shape ``(B, num_kv_heads, T, head_dim)`` and are stored
    *after* RoPE has been applied to the keys, so a cached token never needs to
    be re-processed. Because only ``num_kv_heads`` heads are stored, GQA/MQA
    shrink the cache by ``num_heads / num_kv_heads``.
    """

    def __init__(self, num_layers: int) -> None:
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new ``k``/``v`` for a layer and return the full tensors."""
        old, old_v = self.keys[layer_idx], self.values[layer_idx]
        if old is None or old_v is None:
            self.keys[layer_idx], self.values[layer_idx] = k, v
            return k, v
        else:
            if k.shape[0] != old.shape[0] or k.shape[1] != old.shape[1] or k.shape[3] != old.shape[3]:
                raise ValueError(
                    f"cache holds keys of shape {tuple(old.shape)} but got {tuple(k.shape)}: "
                    "batch size, KV heads and head_dim must match (call cache.reset() to start a new batch)"
                )
            new_k, new_v = torch.cat([old, k], dim=2), torch.cat([old_v, v], dim=2)
            self.keys[layer_idx], self.values[layer_idx] = new_k, new_v
            return new_k, new_v

    def reset(self) -> None:
        """Drop all cached keys/values so the cache can be reused for a new batch."""
        n = len(self.keys)
        self.keys, self.values = [None] * n, [None] * n

    @property
    def seq_len(self) -> int:
        """Number of cached positions (taken from layer 0)."""
        return 0 if self.keys[0] is None else self.keys[0].shape[2]

    def nbytes(self) -> int:
        """Actual memory held by the cache in bytes."""
        return sum(t.numel() * t.element_size() for t in (*self.keys, *self.values) if t is not None)
