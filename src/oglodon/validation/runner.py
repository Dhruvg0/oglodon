"""``validate_model``: run the registered checks and build a report."""

from __future__ import annotations

import copy
import time
import traceback
from collections.abc import Iterable, Sequence

import torch

from ..config import OglodonConfig
from . import checks_attention, checks_cache, checks_model, checks_training  # noqa: F401  (register checks)
from .registry import (
    CATEGORIES,
    DEFAULT_TOLERANCES,
    REGISTRY,
    Ctx,
    Outcome,
    Skip,
    Unsupported,
    make_stress_model,
)
from .report import CheckResult, Status, ValidationReport, collect_metadata

_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _resolve_dtype(dtype: str | torch.dtype | None, model) -> torch.dtype:
    if dtype is None:
        return next(model.parameters()).dtype
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    return _DTYPES[dtype]


def validate_model(
    model,
    config: OglodonConfig | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    seed: int = 0,
    categories: Iterable[str] | None = None,
    select: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    tokenizer: object | None = None,
    tolerances: dict[str, float] | None = None,
) -> ValidationReport:
    """Run behavioural validation on ``model`` and return a :class:`ValidationReport`.

    The caller's model is never modified: every check works on deep copies (the weights as given,
    plus a "stress" copy with large random weights of the same architecture, see
    :func:`oglodon.validation.registry.make_stress_model`).

    Args:
        model: a :class:`oglodon.Oglodon`.
        config: the config the model is *supposed* to have; a mismatch with ``model.config`` is a failure.
        device / dtype: where/in what precision to run (default: the model's own). Requesting CUDA when
            it is not available raises ``ValueError`` instead of silently running on the CPU.
        seed: seeds all random inputs (reported in the metadata).
        categories: subset of ``attention``, ``kv_cache``, ``model``, ``training`` (default: all).
        select / exclude: check names or name prefixes to run / not run. Excluded checks are reported as
            ``skipped``, never as passed.
        tokenizer: optional, enables the tokenizer/vocabulary compatibility check.
        tolerances: override ``{"ref": ..., "equiv": ...}`` (reference-comparison and path-equivalence tolerances).
    """
    from ..quantization import is_quantized

    if is_quantized(model):
        raise ValueError(
            "validate_model compares float weights with a float64 reference and cannot check an int8 model; "
            "use `oglodon verify --checkpoint DIR` (behavioural checks) and `oglodon quantize --text` (drift report)"
        )
    cats = tuple(CATEGORIES if categories is None else categories)
    bad = [c for c in cats if c not in CATEGORIES]
    if bad:
        raise ValueError(f"unknown categories {bad}; choose from {list(CATEGORIES)}")
    dev = torch.device(device) if device is not None else model.device
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available in this process")
    dt = _resolve_dtype(dtype, model)

    target = copy.deepcopy(model).to(dev).to(dt).eval()
    cfg = config if config is not None else target.config
    tol = {
        **DEFAULT_TOLERANCES.get(str(dt).replace("torch.", ""), DEFAULT_TOLERANCES["float32"]),
        **(tolerances or {}),
    }
    ctx = Ctx(
        model=target,
        stress=make_stress_model(target, seed),
        cfg=target.config,
        device=dev,
        dtype=dt,
        seed=seed,
        tol=tol,
        tokenizer=tokenizer,
        requested_config=cfg,
        gen=torch.Generator().manual_seed(seed),
    )

    meta = collect_metadata(target.config.to_dict(), dev, dt, seed)
    meta["categories"] = list(cats)
    meta["tolerances"] = tol
    report = ValidationReport(metadata=meta)

    for spec in REGISTRY:
        if spec.category not in cats:
            continue
        res = CheckResult(spec.name, spec.category, Status.SKIPPED, device=str(dev), dtype=ctx.dtype_name)
        wanted = select is None or any(spec.name.startswith(s) for s in select)
        if not wanted:
            continue
        if any(spec.name.startswith(e) for e in exclude):
            res.detail = "excluded by the caller"
            report.checks.append(res)
            continue
        t0 = time.perf_counter()
        try:
            out = spec.fn(ctx)
        except Skip as e:
            out = Outcome(Status.SKIPPED, str(e))
        except Unsupported as e:
            out = Outcome(Status.UNSUPPORTED, str(e))
        except Exception as e:  # a crashing check is a failing check, never a pass
            tb = traceback.extract_tb(e.__traceback__)[-1]
            out = Outcome(
                Status.FAIL, f"crashed: {type(e).__name__}: {e} ({tb.filename.split('/')[-1]}:{tb.lineno})"
            )
        res.status, res.detail = out.status, out.detail
        res.max_abs_diff, res.tolerance = out.max_abs_diff, out.tolerance
        if out.dtype:
            res.dtype = out.dtype
        res.duration_s = round(time.perf_counter() - t0, 4)
        report.checks.append(res)
    return report
