"""Token and (optional) learned absolute position embeddings."""

from __future__ import annotations

import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    """Trainable absolute position table, used when ``use_rope=False``."""

    def __init__(self, max_seq_len: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dim)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(position_ids)
