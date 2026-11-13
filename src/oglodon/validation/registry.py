"""Check registry and the shared context passed to every check."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from ..config import OglodonConfig
from .report import Status


class Skip(Exception):  # noqa: N818 - control-flow signal, not an error
    """Raised by a check that cannot run because a precondition is missing (-> SKIPPED)."""


class Unsupported(Exception):  # noqa: N818
    """Raised by a check that this model/device/torch build cannot run (-> UNSUPPORTED)."""


@dataclass
class Outcome:
    status: Status
    detail: str = ""
    max_abs_diff: float | None = None
    tolerance: float | None = None
    dtype: str | None = None  # override when the check runs in a different precision than the model


# Default tolerances per model dtype. ``ref`` compares against the float64 reference, ``equiv``
# compares two paths that should be numerically identical (e.g. cached vs uncached).
DEFAULT_TOLERANCES: dict[str, dict[str, float]] = {
    "float64": {"ref": 1e-9, "equiv": 1e-10},
    "float32": {"ref": 1e-4, "equiv": 1e-5},
    "bfloat16": {"ref": 1e-1, "equiv": 5e-2},
    "float16": {"ref": 3e-2, "equiv": 1e-2},
}


@dataclass
class Ctx:
    """Everything a check needs. ``model`` keeps the caller's weights; ``stress`` has the same
    architecture with large random weights so that bugs cannot hide behind a near-zero signal."""

    model: Any  # a oglodon.Oglodon (typed Any: checks reach into submodules dynamically)
    stress: Any
    cfg: OglodonConfig
    device: torch.device
    dtype: torch.dtype
    seed: int
    tol: dict[str, float]
    tokenizer: object | None = None
    requested_config: OglodonConfig | None = None  # config the caller says the model should have
    gen: torch.Generator = field(default_factory=torch.Generator)

    @property
    def dtype_name(self) -> str:
        return str(self.dtype).replace("torch.", "")

    def ids(self, batch: int, seq: int) -> torch.Tensor:
        return torch.randint(0, self.cfg.vocab_size, (batch, seq), generator=self.gen).to(self.device)

    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen, dtype=torch.float32).to(self.device, self.dtype)

    def fresh(self, stress: bool = True) -> Any:
        """A new deep copy (so a check can modify it freely)."""
        return copy.deepcopy(self.stress if stress else self.model)

    def numeric(self, diff: float, tol: float, what: str, scale: float = 1.0) -> Outcome:
        """PASS iff ``diff <= tol * max(1, scale)``: ``scale`` is the magnitude of the compared values
        (floating-point error grows with it), the reported tolerance is the effective one."""
        tol = tol * max(1.0, scale)
        ok = math.isfinite(diff) and diff <= tol
        detail = what if ok else f"{what}: difference {diff:.3e} exceeds tolerance {tol:.1e}"
        return Outcome(Status.PASS if ok else Status.FAIL, detail, float(diff), float(tol))

    def exact(self, ok: bool, what: str, count: float | None = None) -> Outcome:
        return Outcome(Status.PASS if ok else Status.FAIL, what, count, 0.0 if count is not None else None)


@dataclass(frozen=True)
class CheckSpec:
    name: str
    category: str
    fn: Callable[[Ctx], Outcome]


REGISTRY: list[CheckSpec] = []
CATEGORIES = ("attention", "kv_cache", "model", "training")


def register(category: str, name: str):
    """Decorator adding ``fn(ctx) -> Outcome`` to the registry."""

    def deco(fn: Callable[[Ctx], Outcome]):
        assert category in CATEGORIES and not any(s.name == name for s in REGISTRY), name
        REGISTRY.append(CheckSpec(name, category, fn))
        return fn

    return deco


def make_stress_model(model: Any, seed: int, std: float = 0.3) -> Any:
    """Same architecture/dtype/device, weights re-drawn large, dropout off.

    At the usual ``init_std=0.02`` attention and the MLP barely influence the logits, so a masking
    or RoPE bug would change them by ~1e-4. With these weights such a bug changes them by ~1e-1.
    """
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed + 12345)
    with torch.no_grad():
        for name, p in m.named_parameters():
            noise = torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.device, p.dtype)
            if p.dim() >= 2:
                p.copy_(noise * std)
            elif name.endswith("bias"):
                p.copy_(noise * 0.1)
            else:  # normalisation gains
                p.copy_(1.0 + 0.1 * noise)
    for mod in m.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    return m.eval()
