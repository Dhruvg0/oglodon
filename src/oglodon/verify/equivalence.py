"""Cross-implementation verification: do two implementations compute the same function?

Invariants (``oglodon.verify``) are necessary but not sufficient: a model can be causal, cache-consistent and
differentiable and still compute the wrong function (for example a consistently wrong attention scale or RoPE
pairing). The strongest evidence is a *second, independent implementation* with identical weights.
``verify_conversion(source, convert)`` converts ``source`` (e.g. a Hugging Face Llama) with ``convert``
(e.g. :func:`oglodon.convert.from_hf_llama`) and compares logits, padded logits, loss and greedy generation,
on the weights you gave and on a large-weight stress copy.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..validation.registry import DEFAULT_TOLERANCES, Outcome
from ..validation.report import CheckResult, Status, ValidationReport, collect_metadata
from .adapter import CausalLMAdapter
from .checks import describe, make_stress


def _logits(model: Any, ids: torch.Tensor, mask: torch.Tensor | None = None, pos: torch.Tensor | None = None):
    return CausalLMAdapter(model).logits(ids, mask, pos)


def _num(diff: float, tol: float, scale: float, what: str) -> Outcome:
    tol = tol * max(1.0, scale)
    ok = diff <= tol
    return Outcome(
        Status.PASS if ok else Status.FAIL,
        what if ok else f"{what}: difference {diff:.3e} exceeds {tol:.1e}",
        diff,
        tol,
    )


def verify_conversion(
    source: nn.Module,
    convert: Callable[[nn.Module], nn.Module],
    *,
    seed: int = 0,
    seq_len: int = 16,
    tolerance: float | None = None,
) -> ValidationReport:
    """Compare ``source`` with ``convert(source)`` (two implementations, same weights).

    A conversion that is not possible for this architecture (``NotImplementedError``) yields one ``skipped``
    result with the reason, never a pass.
    """
    src = copy.deepcopy(source).eval()
    first = next(src.parameters())
    dt = first.dtype
    tol = (
        tolerance
        or DEFAULT_TOLERANCES.get(str(dt).replace("torch.", ""), DEFAULT_TOLERANCES["float32"])["ref"]
    )
    meta = collect_metadata(describe(src), first.device, dt, seed)
    meta.update(kind="verify_conversion", tolerance=tol)
    report = ValidationReport(metadata=meta)
    vocab = getattr(getattr(src, "config", None), "vocab_size", None)
    gen = torch.Generator().manual_seed(seed)

    def add(name: str, fn: Callable[[], Outcome]) -> None:
        res = CheckResult(
            name, "equivalence", Status.SKIPPED, device=str(first.device), dtype=str(dt).replace("torch.", "")
        )
        t0 = time.perf_counter()
        try:
            out = fn()
        except NotImplementedError as e:
            out = Outcome(Status.SKIPPED, f"conversion not available: {e}")
        except Exception as e:
            out = Outcome(Status.FAIL, f"crashed: {type(e).__name__}: {e}")
        res.status, res.detail, res.max_abs_diff, res.tolerance = (
            out.status,
            out.detail,
            out.max_abs_diff,
            out.tolerance,
        )
        res.duration_s = round(time.perf_counter() - t0, 4)
        report.checks.append(res)

    if vocab is None:
        add("equivalence.setup", lambda: Outcome(Status.SKIPPED, "source model has no config.vocab_size"))
        return report

    def pair(stress: bool):
        s = make_stress(src, seed) if stress else src
        return s, convert(s).eval()

    def ids(b: int, t: int) -> torch.Tensor:
        return torch.randint(0, vocab, (b, t), generator=gen)

    def logits_check(stress: bool) -> Outcome:
        a, b = pair(stress)
        x = ids(3, seq_len)
        with torch.no_grad():
            la, lb = _logits(a, x), _logits(b, x)
        label = "stress" if stress else "given"
        return _num(
            float((la - lb).abs().max()),
            tol,
            float(la.abs().max()),
            f"[{label} weights] logits of both implementations agree",
        )

    def padded_check() -> Outcome:
        a, b = pair(True)
        real, P = ids(2, seq_len), 3
        full = torch.cat([ids(2, P), real], 1)
        mask = torch.cat([torch.zeros(2, P), torch.ones(2, seq_len)], 1).long()
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            la = _logits(a, full, mask, pos)[:, P:]
            lb = _logits(b, full, mask, pos)[:, P:]
        return _num(
            float((la - lb).abs().max()),
            tol,
            float(la.abs().max()),
            "[stress weights] left-padded logits agree at real tokens",
        )

    def loss_check() -> Outcome:
        a, b = pair(True)
        x = ids(3, seq_len)
        labels = x.clone()
        labels[0, : seq_len // 3] = -100
        with torch.no_grad():
            la, lb = _logits(a, x).double(), _logits(b, x).double()

        def ce(lg: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100
            )

        d = abs(float(ce(la)) - float(ce(lb)))
        return _num(d, tol, abs(float(ce(la))), "[stress weights] next-token loss agrees")

    def generation_check() -> Outcome:
        a, b = pair(True)
        x = ids(2, 6)
        n = 10
        with torch.no_grad():
            ga = (
                a.generate(x, max_new_tokens=n, do_sample=False)[:, x.shape[1] :]
                if hasattr(a, "generation_config")
                else None
            )
            gb = b.generate(x, max_new_tokens=n, eos_token_id=None)[:, x.shape[1] :]
        if ga is None:
            raise NotImplementedError("source model has no Hugging Face-style generate()")
        if torch.equal(ga, gb):
            return Outcome(Status.PASS, f"greedy {n}-token continuations (cached) are identical")
        # Free-running greedy decoding amplifies a tiny numerical difference once two tokens are nearly tied (this
        # is common in bfloat16 and with large random weights). So check teacher-forced instead: on the tokens the
        # source model generated, both implementations must pick the same next token wherever the source's top-1
        # lead over top-2 is larger than the logit tolerance.
        full = torch.cat([x, ga.to(x.device)], dim=1)
        with torch.no_grad():
            la, lb = _logits(a, full).double(), _logits(b, full).double()
        limit = tol * max(1.0, float(la.abs().max()))
        top2 = la.topk(2, dim=-1).values
        decisive = (top2[..., 0] - top2[..., 1]) > 2 * limit
        decisive = decisive[:, x.shape[1] - 1 : -1]
        disagree = (la.argmax(-1) != lb.argmax(-1))[:, x.shape[1] - 1 : -1] & decisive
        ties = int((~decisive).sum())
        if bool(disagree.any()):
            return Outcome(
                Status.FAIL,
                f"the implementations pick different tokens where the choice is clear: {int(disagree.sum())} of "
                f"{decisive.numel()} generated positions (continuations {ga.tolist()} vs {gb.tolist()})",
            )
        return Outcome(
            Status.PASS,
            f"free-running greedy continuations differ, but only after near-ties (top-1 lead below twice the "
            f"logit tolerance); both implementations agree on all {int(decisive.sum())} clear positions "
            f"({ties} near-tied positions not compared)",
        )

    add("equivalence.logits_given_weights", lambda: logits_check(False))
    add("equivalence.logits_stress_weights", lambda: logits_check(True))
    add("equivalence.padded_logits", padded_check)
    add("equivalence.loss", loss_check)
    add("equivalence.greedy_generation", generation_check)
    return report
