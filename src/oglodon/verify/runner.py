"""``verify_causal_lm``: run the black-box checks on any causal language model."""

from __future__ import annotations

import copy
import time
import traceback
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn

from ..validation.registry import DEFAULT_TOLERANCES, Outcome, Skip, Unsupported
from ..validation.report import CheckResult, Status, ValidationReport, collect_metadata
from .adapter import CausalLMAdapter, adapt
from .checks import CATEGORIES, VREGISTRY, VCtx, describe, make_stress

_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def verify_causal_lm(
    model: Any,
    *,
    adapter: CausalLMAdapter | None = None,
    vocab_size: int | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    seed: int = 0,
    seq_len: int | None = None,
    categories: Iterable[str] | None = None,
    select: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    tolerances: dict[str, float] | None = None,
    stress: bool = True,
    max_fp64_params: int | None = 250_000_000,
) -> ValidationReport:
    """Verify that ``model`` behaves like a correct causal language model, whatever its implementation.

    ``model`` can be a :class:`oglodon.Oglodon`, a Hugging Face ``*ForCausalLM``, or any ``nn.Module`` whose forward
    maps ``input_ids`` ``(B, T)`` to logits ``(B, T, V)`` (a tensor, an object with ``.logits``, a dict or a tuple).
    Pass ``adapter=CausalLMAdapter(model, forward=...)`` for unusual signatures.

    Memory: the model is deep-copied once (plus once more for the stress weights). Checks that need a float64
    copy (float64/bfloat16 comparison, finite-difference gradients, ...) report *unsupported* above
    ``max_fp64_params`` parameters (default 250M; ``None`` disables the limit). ``stress=False`` skips the
    large-weight copy and runs every check on the weights as given (use for very large models).

    The model is never modified (checks work on deep copies). Results use the same statuses as
    :func:`oglodon.validation.validate_model`; skipped and unsupported checks are never reported as passed.
    """
    cats = tuple(CATEGORIES if categories is None else categories)
    bad = [c for c in cats if c not in CATEGORIES]
    if bad:
        raise ValueError(f"unknown categories {bad}; choose from {list(CATEGORIES)}")
    ad0 = adapter or adapt(model, vocab_size=vocab_size)
    inner = ad0.model
    if not isinstance(inner, nn.Module):
        raise TypeError("verify_causal_lm needs an nn.Module (wrap plain functions in an nn.Module)")
    first = next(inner.parameters(), None)
    dev = (
        torch.device(device)
        if device is not None
        else (first.device if first is not None else torch.device("cpu"))
    )
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available in this process")
    if dtype is None:
        dt = first.dtype if first is not None else torch.float32
    elif isinstance(dtype, torch.dtype):
        dt = dtype
    elif dtype in _DTYPES:
        dt = _DTYPES[dtype]
    else:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")

    target = copy.deepcopy(inner).to(dev).to(dt).eval()
    ad = ad0.with_model(target)
    vocab = ad.infer_vocab(dev)
    T = seq_len or min(ad.max_positions or 16, 16)
    if T < 4:
        raise ValueError("seq_len must be at least 4")
    tol = {
        **DEFAULT_TOLERANCES.get(str(dt).replace("torch.", ""), DEFAULT_TOLERANCES["float32"]),
        **(tolerances or {}),
    }
    ctx = VCtx(
        adapter=ad,
        stress=ad.with_model(make_stress(target, seed)) if stress else ad,
        use_stress=stress,
        fp64_limit=max_fp64_params,
        device=dev,
        dtype=dt,
        vocab=vocab,
        seq_len=T,
        tol=tol,
        gen=torch.Generator().manual_seed(seed),
        extra={"seed": seed},
    )
    cfg = getattr(inner, "config", None)
    cfg_dict = cfg.to_dict() if cfg is not None and hasattr(cfg, "to_dict") else {}
    meta = collect_metadata(
        {**describe(inner), "config": cfg_dict, "vocab_size": vocab, "seq_len": T}, dev, dt, seed
    )
    meta.update(describe(inner))
    meta.update(
        kind="verify_causal_lm",
        categories=list(cats),
        tolerances=tol,
        stress_weights=stress,
        max_fp64_params=max_fp64_params,
    )
    meta["capabilities"] = {
        "attention_mask": ad.supports_mask,
        "position_ids": ad.supports_positions,
        "kv_cache": ad.supports_cache,
        "labels": ad.supports_labels,
    }
    report = ValidationReport(metadata=meta)
    for spec in VREGISTRY:
        if spec.category not in cats:
            continue
        if select is not None and not any(spec.name.startswith(s) for s in select):
            continue
        res = CheckResult(spec.name, spec.category, Status.SKIPPED, device=str(dev), dtype=ctx.dtype_name)
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
        except Exception as e:
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
