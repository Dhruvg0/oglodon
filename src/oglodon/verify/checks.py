"""Black-box behavioural checks for any causal language model.

None of these needs the model's internals or a reference implementation. They test *invariants that every
correct causal LM must satisfy* (metamorphic testing): change a future token and the past must not move,
batch a sequence with others and its logits must not change, pad it and (with an attention mask) the real
tokens must not notice, decode with a KV cache and get the same logits, and so on. Each check is run on the
weights as given and, where it matters, on a "stress" copy with large random weights, because a near-zero
initialisation makes attention almost uniform and hides masking bugs.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ..validation.registry import Outcome, Skip, Unsupported
from ..validation.report import Status
from .adapter import CausalLMAdapter, count_parameters

CATEGORIES = ("structure", "causality", "batching", "cache", "numerics", "gradients")


@dataclass
class VCtx:
    adapter: CausalLMAdapter  # the model exactly as given (eval mode, on the target device)
    stress: CausalLMAdapter  # same architecture, large random matrices
    device: torch.device
    dtype: torch.dtype
    vocab: int
    seq_len: int
    tol: dict[str, float]
    gen: torch.Generator
    pad_id: int = 0
    use_stress: bool = True
    fp64_limit: int | None = None  # None: no limit; else max parameters for float64 copies
    extra: dict = field(default_factory=dict)

    @property
    def dtype_name(self) -> str:
        return str(self.dtype).replace("torch.", "")

    def ids(self, batch: int, seq: int | None = None) -> torch.Tensor:
        return torch.randint(0, self.vocab, (batch, seq or self.seq_len), generator=self.gen).to(self.device)

    def variants(self) -> list[tuple[str, CausalLMAdapter]]:
        if not self.use_stress:
            return [("given weights", self.adapter)]
        return [("given weights", self.adapter), ("stress weights", self.stress)]

    def require_copies(self, what: str) -> None:
        """Raise Unsupported when a full-precision copy of the model would exceed the configured size limit."""
        n = count_parameters(self.adapter.model)
        if self.fp64_limit is not None and n is not None and n > self.fp64_limit:
            raise Unsupported(
                f"{what} needs a float64 copy of the model; it has {n:,} parameters (limit {self.fp64_limit:,}). "
                "Raise max_fp64_params / --max-fp64-params if you have the memory"
            )

    def numeric(self, diff: float, tol: float, what: str, scale: float = 1.0) -> Outcome:
        tol = tol * max(1.0, scale)
        ok = math.isfinite(diff) and diff <= tol
        detail = what if ok else f"{what}: difference {diff:.3e} exceeds tolerance {tol:.1e}"
        return Outcome(Status.PASS if ok else Status.FAIL, detail, float(diff), float(tol))


@dataclass
class VSpec:
    name: str
    category: str
    fn: Callable[[VCtx], Outcome]
    doc: str


VREGISTRY: list[VSpec] = []


def vcheck(name: str, category: str):
    def deco(fn):
        VREGISTRY.append(VSpec(name, category, fn, (fn.__doc__ or "").strip().split("\n")[0]))
        return fn

    return deco


def make_stress(model: nn.Module, seed: int, std: float = 0.2) -> nn.Module:
    """Copy of ``model`` whose matrices (and embeddings) are re-drawn from ``N(0, std)``.

    Biases are drawn from ``N(0, 0.1)`` (they start at zero almost everywhere, which hides bias bugs);
    norm gains are kept, so normalisation layers still behave. This is only a
    way to make hidden bugs *visible*; it makes no claim about trained behaviour."""
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed + 12345)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() >= 2 and p.is_floating_point():
                p.copy_((torch.randn(p.shape, generator=g) * std).to(p.device, p.dtype))
            elif p.dim() == 1 and name.endswith("bias") and p.is_floating_point():
                # biases start at zero in almost every library, which would hide bias-handling bugs
                p.copy_((torch.randn(p.shape, generator=g) * 0.1).to(p.device, p.dtype))
    return m.eval()


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


def _scale(*ts: torch.Tensor) -> float:
    return max(float(t.detach().abs().max()) for t in ts)


def _manual_loss(logits: torch.Tensor, labels: torch.Tensor, shift: bool = True) -> torch.Tensor:
    lg = logits.double()
    if shift:
        lg, labels = lg[:, :-1], labels[:, 1:]
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), labels.reshape(-1), ignore_index=-100)


def _combine(results: list[Outcome]) -> Outcome:
    """Worst status wins; numeric fields come from the worst numeric result."""
    order = {Status.FAIL: 3, Status.WARNING: 2, Status.UNSUPPORTED: 1, Status.PASS: 0, Status.SKIPPED: 0}
    worst = max(results, key=lambda r: order[r.status])
    nums = [r for r in results if r.max_abs_diff is not None]
    top = max(nums, key=lambda r: (r.max_abs_diff or 0.0) / max(r.tolerance or 1.0, 1e-30)) if nums else worst
    return Outcome(worst.status, "; ".join(r.detail for r in results), top.max_abs_diff, top.tolerance)


# ================================================================================ structure
@vcheck("structure.shape_and_vocab", "structure")
def shape_and_vocab(ctx: VCtx) -> Outcome:
    """Logits are (B, T, V) with V equal to the declared vocabulary; extreme token ids work; outputs finite."""
    ad = ctx.adapter
    with torch.no_grad():
        ids = ctx.ids(3)
        lg = ad.logits(ids)
        if tuple(lg.shape[:2]) != tuple(ids.shape):
            return Outcome(
                Status.FAIL, f"logits shape {tuple(lg.shape)} does not start with (B,T)={tuple(ids.shape)}"
            )
        if lg.shape[-1] != ctx.vocab:
            return Outcome(Status.FAIL, f"logits width {lg.shape[-1]} != vocab_size {ctx.vocab}")
        edge = torch.tensor([[0, ctx.vocab - 1] * (ctx.seq_len // 2)], device=ctx.device)
        e = ad.logits(edge)
        if not (torch.isfinite(lg).all() and torch.isfinite(e).all()):
            return Outcome(Status.FAIL, "non-finite logits for ordinary or extreme (0 / vocab-1) token ids")
        one = ad.logits(ids[:, :1])
        if one.shape[1] != 1 or not torch.isfinite(one).all():
            return Outcome(Status.FAIL, "a length-1 input does not give finite (B,1,V) logits")
    return Outcome(Status.PASS, f"logits (B,T,{ctx.vocab}), finite, length-1 and extreme-id inputs fine")


@vcheck("structure.determinism", "structure")
def determinism(ctx: VCtx) -> Outcome:
    """In eval mode two identical forward passes give identical logits (no hidden randomness)."""
    res = []
    with torch.no_grad():
        ids = ctx.ids(2)
        for label, ad in ctx.variants():
            a, b = ad.logits(ids), ad.logits(ids)
            res.append(
                ctx.numeric(
                    _maxdiff(a, b), ctx.tol["equiv"], f"[{label}] repeated eval passes agree", _scale(a)
                )
            )
    return _combine(res)


@vcheck("structure.state_dict_roundtrip", "structure")
def state_dict_roundtrip(ctx: VCtx) -> Outcome:
    """Loading the state dict into a re-initialised copy reproduces the logits exactly."""
    m = ctx.adapter.model
    if not isinstance(m, nn.Module):
        raise Skip("model is not an nn.Module")
    other = make_stress(m, ctx.extra.get("seed", 0) + 1)
    other.load_state_dict(m.state_dict(), strict=True)
    ids = ctx.ids(2)
    with torch.no_grad():
        a, b = ctx.adapter.logits(ids), ctx.adapter.with_model(other.eval()).logits(ids)
    return ctx.numeric(
        _maxdiff(a, b), ctx.tol["equiv"], "state_dict -> fresh copy reproduces logits", _scale(a)
    )


@vcheck("structure.max_length", "structure")
def max_length(ctx: VCtx) -> Outcome:
    """The model runs, with finite output, at its maximum supported length."""
    mx = ctx.adapter.max_positions
    if not mx:
        raise Skip("model does not declare a maximum length")
    n = min(mx, 512)
    with torch.no_grad():
        lg = ctx.adapter.logits(ctx.ids(1, n))
    ok = bool(torch.isfinite(lg).all())
    return Outcome(
        Status.PASS if ok else Status.FAIL,
        f"length {n} (of max {mx}) gives {'finite' if ok else 'NON-FINITE'} logits",
    )


# ================================================================================ causality
@vcheck("causality.future_independence", "causality")
def future_independence(ctx: VCtx) -> Outcome:
    """Changing token t leaves the logits at positions < t unchanged (and does change positions >= t)."""
    T = ctx.seq_len
    res = []
    for label, ad in ctx.variants():
        with torch.no_grad():
            ids = ctx.ids(2, T)
            base = ad.logits(ids)
            worst, moved = 0.0, 0.0
            for t in sorted({1, T // 2, T - 1}):
                alt = ids.clone()
                alt[:, t] = (
                    alt[:, t] + 1 + int(torch.randint(0, max(ctx.vocab - 1, 1), (1,), generator=ctx.gen))
                ) % ctx.vocab
                out = ad.logits(alt)
                worst = max(worst, _maxdiff(base[:, :t], out[:, :t]))
                moved = max(moved, _maxdiff(base[:, t:], out[:, t:]))
        if moved < 1e-8:
            res.append(
                Outcome(
                    Status.FAIL,
                    f"[{label}] logits ignore the input entirely (changing a token moved nothing), so causality cannot be tested",
                )
            )
            continue
        res.append(
            ctx.numeric(
                worst,
                ctx.tol["equiv"],
                f"[{label}] earlier logits unaffected by later tokens (later ones moved by {moved:.1e})",
                _scale(base),
            )
        )
    return _combine(res)


@vcheck("causality.prefix_consistency", "causality")
def prefix_consistency(ctx: VCtx) -> Outcome:
    """Logits for a prefix equal the first rows of the logits for the full sequence."""
    res = []
    for label, ad in ctx.variants():
        with torch.no_grad():
            ids = ctx.ids(2)
            full = ad.logits(ids)
            k = max(ctx.seq_len // 3, 1)
            part = ad.logits(ids[:, :k])
        res.append(
            ctx.numeric(
                _maxdiff(full[:, :k], part),
                ctx.tol["equiv"],
                f"[{label}] prefix of length {k} == head of full pass",
                _scale(full),
            )
        )
    return _combine(res)


@vcheck("causality.padding_invariance", "causality")
def padding_invariance(ctx: VCtx) -> Outcome:
    """Left padding with an attention mask (and mask-derived positions) does not change real-token logits."""
    ad = ctx.adapter
    if not ad.supports_mask:
        raise Skip("model has no attention_mask argument")
    if not ad.supports_positions:
        raise Skip(
            "model has no position_ids argument: with left padding positions legitimately shift, so invariance cannot be asked of it"
        )
    res = []
    P, T = 3, ctx.seq_len
    for label, a in ctx.variants():
        with torch.no_grad():
            real = ctx.ids(2, T)
            pads = ctx.ids(2, P)  # random tokens in the pad slots: if the mask is ignored they are visible
            ids = torch.cat([pads, real], dim=1)
            mask = torch.cat([torch.zeros(2, P), torch.ones(2, T)], dim=1).long().to(ctx.device)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            padded = a.logits(ids, attention_mask=mask, position_ids=pos)[:, P:]
            plain = a.logits(
                real,
                attention_mask=torch.ones_like(real),
                position_ids=torch.arange(T, device=ctx.device).expand(2, T),
            )
        if not torch.isfinite(padded).all():
            res.append(Outcome(Status.FAIL, f"[{label}] padding produced non-finite logits at real tokens"))
            continue
        res.append(
            ctx.numeric(
                _maxdiff(padded, plain),
                ctx.tol["equiv"],
                f"[{label}] {P} left-pad tokens do not change real-token logits",
                _scale(plain),
            )
        )
    return _combine(res)


@vcheck("causality.mask_is_used", "causality")
def mask_is_used(ctx: VCtx) -> Outcome:
    """Masked-out (0) positions are really invisible: changing them changes nothing at real positions."""
    ad = ctx.adapter
    if not ad.supports_mask:
        raise Skip("model has no attention_mask argument")
    res = []
    P, T = 3, ctx.seq_len
    for label, a in ctx.variants():
        with torch.no_grad():
            real = ctx.ids(2, T)
            mask = torch.cat([torch.zeros(2, P), torch.ones(2, T)], dim=1).long().to(ctx.device)
            pos = (mask.cumsum(-1) - 1).clamp(min=0) if ad.supports_positions else None
            x1 = torch.cat([ctx.ids(2, P), real], dim=1)
            x2 = torch.cat([ctx.ids(2, P), real], dim=1)
            o1 = a.logits(x1, attention_mask=mask, position_ids=pos)[:, P:]
            o2 = a.logits(x2, attention_mask=mask, position_ids=pos)[:, P:]
        res.append(
            ctx.numeric(
                _maxdiff(o1, o2),
                ctx.tol["equiv"],
                f"[{label}] contents of masked slots are invisible",
                _scale(o1),
            )
        )
    return _combine(res)


# ================================================================================ batching
@vcheck("batching.sample_independence", "batching")
def sample_independence(ctx: VCtx) -> Outcome:
    """Each row of a batch gets the same logits as when run alone (no cross-sample leakage)."""
    res = []
    for label, ad in ctx.variants():
        with torch.no_grad():
            ids = ctx.ids(4)
            batch = ad.logits(ids)
            alone = torch.cat([ad.logits(ids[i : i + 1]) for i in range(4)])
        res.append(
            ctx.numeric(
                _maxdiff(batch, alone), ctx.tol["equiv"], f"[{label}] batched == one-at-a-time", _scale(batch)
            )
        )
    return _combine(res)


@vcheck("batching.order_independence", "batching")
def order_independence(ctx: VCtx) -> Outcome:
    """Permuting the batch permutes the logits in the same way."""
    res = []
    for label, ad in ctx.variants():
        with torch.no_grad():
            ids = ctx.ids(4)
            perm = torch.tensor([2, 0, 3, 1], device=ctx.device)
            a, b = ad.logits(ids)[perm], ad.logits(ids[perm])
        res.append(
            ctx.numeric(
                _maxdiff(a, b),
                ctx.tol["equiv"],
                f"[{label}] batch permutation commutes with the model",
                _scale(a),
            )
        )
    return _combine(res)


# ================================================================================ cache
def _cache_check(ctx: VCtx, chunk: int, what: str) -> Outcome:
    ad = ctx.adapter
    if not ad.supports_cache:
        raise Skip("model has no use_cache / past_key_values interface")
    res = []
    for label, a in ctx.variants():
        with torch.no_grad():
            ids = ctx.ids(2)
            full = a.logits(ids)
            k = max(ctx.seq_len // 2, 1)
            try:
                cached = a.cached_logits(ids, prefill=k, chunk=chunk)
            except NotImplementedError as e:
                raise Skip(str(e)) from e
            except TypeError as e:
                raise Unsupported(f"cache interface is not usable through the standard call: {e}") from e
        res.append(
            ctx.numeric(
                _maxdiff(full, cached),
                ctx.tol["equiv"] * 4,
                f"[{label}] {what} == full forward",
                _scale(full),
            )
        )
    return _combine(res)


@vcheck("cache.single_token_steps", "cache")
def cache_single_steps(ctx: VCtx) -> Outcome:
    """Prefill + one-token-at-a-time decoding with the KV cache reproduces the full-forward logits."""
    return _cache_check(ctx, 1, "prefill + single-token cached steps")


@vcheck("cache.chunked_steps", "cache")
def cache_chunked(ctx: VCtx) -> Outcome:
    """Prefill + multi-token chunks with the KV cache reproduces the full-forward logits (cache offset in the mask)."""
    return _cache_check(ctx, 3, "prefill + 3-token cached chunks")


# ================================================================================ numerics
@vcheck("numerics.float64_consistency", "numerics")
def float64_consistency(ctx: VCtx) -> Outcome:
    """A float64 copy agrees with the native-precision logits (large gaps mean unstable or dtype-fragile maths)."""
    ctx.require_copies("float64 consistency")
    m = ctx.stress.model
    if not isinstance(m, nn.Module):
        raise Skip("model is not an nn.Module")
    ids = ctx.ids(2)
    with torch.no_grad():
        low = ctx.stress.logits(ids)
        try:
            hi = ctx.stress.with_model(copy.deepcopy(m).double().eval()).logits(ids)
        except Exception as e:
            raise Unsupported(
                f"model cannot run in float64 ({type(e).__name__}: {e}); portability limit, not proven wrong"
            ) from e
    tol = {"float64": 1e-9, "float32": 2e-3, "bfloat16": 0.3, "float16": 0.1}.get(ctx.dtype_name, 2e-3)
    return ctx.numeric(
        _maxdiff(low, hi), tol, f"{ctx.dtype_name} logits vs float64 copy of the same weights", _scale(hi)
    )


@vcheck("numerics.bfloat16_sanity", "numerics")
def bfloat16_sanity(ctx: VCtx) -> Outcome:
    """A bfloat16 copy runs, stays finite and stays near the float64 answer (numerical-risk warning)."""
    ctx.require_copies("bfloat16 sanity")
    m = ctx.stress.model
    if not isinstance(m, nn.Module):
        raise Skip("model is not an nn.Module")
    ids = ctx.ids(2)
    with torch.no_grad():
        try:
            ref = ctx.stress.with_model(copy.deepcopy(m).double().eval()).logits(ids)
            lo = ctx.stress.with_model(copy.deepcopy(m).to(torch.bfloat16).eval()).logits(ids)
        except Exception as e:
            raise Unsupported(
                f"bfloat16 or float64 execution failed on this build ({type(e).__name__}: {e})"
            ) from e
    if not torch.isfinite(lo).all():
        return Outcome(Status.FAIL, "bfloat16 copy produces non-finite logits")
    rel = _maxdiff(lo, ref) / max(_scale(ref), 1e-9)
    st = Status.PASS if rel <= 0.25 else Status.WARNING
    return Outcome(
        st,
        f"bfloat16 max error is {rel:.1%} of the logit scale"
        + ("" if st is Status.PASS else " (numerically fragile in half precision)"),
        rel,
        0.25,
    )


@vcheck("numerics.large_activation_stability", "numerics")
def large_activation_stability(ctx: VCtx) -> Outcome:
    """Scaling all weights up 3x (large activations) still gives finite logits (softmax / norm stability)."""
    ctx.require_copies("the large-activation check")
    m = ctx.stress.model
    if not isinstance(m, nn.Module):
        raise Skip("model is not an nn.Module")
    big = copy.deepcopy(m)
    with torch.no_grad():
        for p in big.parameters():
            if p.dim() >= 2:
                p.mul_(3.0)
    with torch.no_grad():
        lg = ctx.stress.with_model(big.eval()).logits(ctx.ids(2))
    ok = bool(torch.isfinite(lg).all())
    return Outcome(
        Status.PASS if ok else Status.FAIL,
        "finite logits with 3x-scaled weights" if ok else "NaN/Inf with 3x-scaled weights",
    )


# ================================================================================ gradients
def _grad_setup(ctx: VCtx, ad: CausalLMAdapter):
    ids = ctx.ids(2)
    return ids, ids.clone()


@vcheck("gradients.finite_and_flowing", "gradients")
def grads_flow(ctx: VCtx) -> Outcome:
    """Backprop gives finite gradients; parameters that receive none (or exactly zero) are listed."""
    ctx.require_copies("the gradient check")
    m = ctx.stress.model
    if not isinstance(m, nn.Module):
        raise Skip("model is not an nn.Module")
    mm = copy.deepcopy(m).train(False)
    ids, labels = _grad_setup(ctx, ctx.stress)
    mm.zero_grad(set_to_none=True)
    _manual_loss(ctx.stress.with_model(mm).logits(ids), labels).backward()
    bad, dead = [], []
    for n, p in mm.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            dead.append(n)
        elif not torch.isfinite(p.grad).all():
            bad.append(n)
        elif float(p.grad.abs().max()) == 0.0:
            dead.append(n)
    if bad:
        return Outcome(Status.FAIL, f"non-finite gradients in {bad[:5]}")
    if dead:
        return Outcome(
            Status.WARNING, f"{len(dead)} parameter tensor(s) get no / exactly-zero gradient: {dead[:5]}"
        )
    return Outcome(Status.PASS, "all trainable parameters receive finite, non-zero gradients")


@vcheck("gradients.finite_difference", "gradients")
def finite_difference(ctx: VCtx) -> Outcome:
    """Autograd gradients match central finite differences in float64 (catches wrong/broken backward passes)."""
    ctx.require_copies("the finite-difference check")
    m = ctx.stress.model
    if not isinstance(m, nn.Module):
        raise Skip("model is not an nn.Module")
    try:
        md = copy.deepcopy(m).double().eval()
    except Exception as e:
        raise Unsupported(f"cannot convert to float64: {e}") from e
    ad = ctx.stress.with_model(md)
    ids = ctx.ids(2, min(ctx.seq_len, 12))
    labels = ids.clone()
    labels[:, 0] = -100

    def loss() -> torch.Tensor:
        return _manual_loss(ad.logits(ids), labels)

    try:
        md.zero_grad(set_to_none=True)
        loss().backward()
    except Exception as e:
        raise Unsupported(f"float64 backward failed: {type(e).__name__}: {e}") from e
    g = torch.Generator().manual_seed(ctx.extra.get("seed", 0))
    params = [
        (n, p)
        for n, p in md.named_parameters()
        if p.grad is not None and p.requires_grad and float(p.grad.abs().max()) > 0
    ]
    if not params:
        raise Skip("no parameter received a gradient")
    order = torch.randperm(len(params), generator=g).tolist()[:8]
    worst, worst_name = 0.0, ""
    for idx in order:
        name, p = params[idx]
        j = int(p.grad.flatten().abs().argmax())  # the most informative coordinate of this tensor
        an = float(p.grad.flatten()[j])
        best = float("inf")
        # Several step sizes: a small one is best in exact float64, but models that silently compute
        # part of their maths in float32 (e.g. a float32 softmax) need a larger one. A genuinely
        # wrong gradient disagrees for *every* step, so the smallest error is what counts.
        for eps in (1e-6, 1e-4, 1e-3):
            with torch.no_grad():
                v = p.data.view(-1)
                old = float(v[j])
                v[j] = old + eps
                lp = float(loss())
                v[j] = old - eps
                lm = float(loss())
                v[j] = old
            fd = (lp - lm) / (2 * eps)
            best = min(best, abs(fd - an) / max(1.0, abs(an), abs(fd)))
        if best > worst:
            worst, worst_name = best, name
    tol = 1e-3
    ok = worst <= tol
    detail = f"autograd == finite differences on {len(order)} parameter tensors (max rel. error {worst:.1e})"
    if ok and worst > 1e-5:
        detail += "; limited by internal sub-float64 arithmetic in the model, not by the gradient"
    if not ok:
        detail = f"autograd disagrees with finite differences for '{worst_name}': rel. error {worst:.2e} > {tol:.0e}"
    return Outcome(Status.PASS if ok else Status.FAIL, detail, worst, tol, dtype="float64")


@vcheck("gradients.loss_convention", "gradients")
def loss_convention(ctx: VCtx) -> Outcome:
    """The model's own loss (if any) equals shifted next-token cross-entropy over its logits (no off-by-one)."""
    ad = ctx.adapter
    if not ad.supports_labels:
        raise Skip("model does not accept labels")
    res = []
    for label, a in ctx.variants():
        with torch.no_grad():
            ids = ctx.ids(3)
            labels = ids.clone()
            labels[0, : ctx.seq_len // 3] = -100  # exercises ignore_index and uneven token counts
            own = a.loss(ids, labels)
            if own is None:
                raise Skip("model returned no loss for labels=...")
            lg = a.logits(ids)
        shifted = float(_manual_loss(lg, labels, True))
        got = float(own)
        tol = ctx.tol["ref"] * max(1.0, abs(shifted))
        if abs(got - shifted) <= tol:
            res.append(
                Outcome(
                    Status.PASS,
                    f"[{label}] own loss == shifted next-token cross-entropy",
                    abs(got - shifted),
                    tol,
                )
            )
        elif abs(got - float(_manual_loss(lg, labels, False))) <= tol:
            res.append(
                Outcome(
                    Status.FAIL,
                    f"[{label}] loss predicts the SAME position (no label shift): it will train an identity map",
                    abs(got - shifted),
                    tol,
                )
            )
        else:
            res.append(
                Outcome(
                    Status.FAIL,
                    f"[{label}] own loss {got:.5f} != shifted cross-entropy {shifted:.5f}",
                    abs(got - shifted),
                    tol,
                )
            )
    return _combine(res)


def describe(model) -> dict:
    return {"model_class": type(model).__qualname__, "num_parameters": count_parameters(model)}
