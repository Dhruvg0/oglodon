"""Learning-rate schedules (multipliers applied to the base learning rate)."""

from __future__ import annotations

import math
from collections.abc import Callable


def make_lr_lambda(
    kind: str, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1
) -> Callable[[int], float]:
    """Return ``f(step) -> multiplier`` with linear warm-up then ``kind`` decay.

    ``kind`` is ``"cosine"``, ``"linear"`` or ``"constant"``. After warm-up the
    multiplier decays to ``min_lr_ratio`` (not to zero) at ``max_steps``.
    """
    if kind not in ("cosine", "linear", "constant"):
        raise ValueError(f"unknown schedule {kind!r}")
    if warmup_steps < 0 or max_steps <= 0 or warmup_steps > max_steps:
        raise ValueError("need 0 <= warmup_steps <= max_steps and max_steps > 0")

    def f(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if kind == "constant":
            return 1.0
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        if kind == "linear":
            return 1.0 - (1.0 - min_lr_ratio) * progress
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return f
