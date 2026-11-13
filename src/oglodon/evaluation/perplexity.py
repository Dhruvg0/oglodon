"""Perplexity of a causal LM over a token sequence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..verify.adapter import CausalLMAdapter
from ._util import model_device


@dataclass
class PerplexityResult:
    loss: float  # mean negative log-likelihood per scored token (nats)
    perplexity: float
    tokens_scored: int
    windows: int
    block_size: int
    stride: int

    @property
    def bits_per_token(self) -> float:
        return self.loss / math.log(2)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "bits_per_token": self.bits_per_token}


@torch.no_grad()
def perplexity(
    model: Any,
    tokens: Sequence[int] | np.ndarray | torch.Tensor,
    *,
    block_size: int | None = None,
    stride: int | None = None,
    batch_size: int = 8,
    device: str | torch.device | None = None,
) -> PerplexityResult:
    """Perplexity of ``tokens`` under ``model`` with a sliding window.

    Each window holds ``block_size`` tokens; consecutive windows start ``stride`` tokens apart. A token is
    scored once, by the first window in which it has at least one token of context, so ``tokens_scored`` is
    ``len(tokens) - 1`` whenever ``stride < block_size``. With ``stride == block_size`` (the fast,
    non-overlapping estimate) the first token of every later window has no context and is skipped, and tokens
    near a window start see little context; a smaller stride gives every scored token more context and a
    lower, more faithful perplexity at the cost of more compute. Defaults: ``block_size`` = the model's maximum
    length (capped at 1024) and ``stride = block_size``.

    The model is put into eval mode for the call. Works with any model the adapter understands.
    """
    adapter = CausalLMAdapter(model)
    ids = (
        torch.as_tensor(np.asarray(tokens) if not isinstance(tokens, torch.Tensor) else tokens)
        .long()
        .flatten()
    )
    n = ids.numel()
    if n < 2:
        raise ValueError("need at least 2 tokens to score anything")
    limit = adapter.max_positions or 1024
    block = min(block_size or min(limit, 1024), n, limit)
    stride = stride or block
    if not 0 < stride <= block:
        raise ValueError(f"stride must be in (0, block_size={block}], got {stride}")
    dev = model_device(model, device)
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    # window k covers [start, start+block); it scores the targets ids[scored_from+1 .. end]
    windows: list[tuple[int, int, int]] = []
    start, scored_to = 0, 0  # scored_to: index of the last target already scored (token 0 has no prediction)
    while True:
        end = min(start + block, n)
        windows.append((start, end, scored_to))
        scored_to_new = end - 1
        if end == n:
            break
        scored_to = scored_to_new
        start += stride
    total, count = 0.0, 0
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        # group windows of equal length (all but possibly the last) so the batch is rectangular
        by_len: dict[int, list[tuple[int, int, int]]] = {}
        for w in chunk:
            by_len.setdefault(w[1] - w[0], []).append(w)
        for group in by_len.values():
            x = torch.stack([ids[s:e] for s, e, _ in group]).to(dev)
            logits = adapter.logits(x)
            logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
            nll = -logp.gather(-1, x[:, 1:, None]).squeeze(-1)  # (G, length-1): target x[:, j+1]
            for row, (s, _, prev) in enumerate(group):
                # target positions (absolute) s+1 .. s+length-1; skip those already scored (<= prev)
                first_new = max(prev - s, 0)  # absolute target index t is new iff t > prev
                new = nll[row, first_new:]
                total += float(new.sum())
                count += new.numel()
    if hasattr(model, "train"):
        model.train(was_training)
    loss = total / count
    return PerplexityResult(loss, math.exp(min(loss, 700)), count, len(windows), block, stride)
