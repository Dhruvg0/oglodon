"""Small shared helpers."""

from __future__ import annotations

from typing import Any

import torch


def model_device(model: Any, device: str | torch.device | None = None) -> torch.device:
    """``device`` if given, else the device of the model's first parameter, else CPU (parameter-less callables)."""
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")
