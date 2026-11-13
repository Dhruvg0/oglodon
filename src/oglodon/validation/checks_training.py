"""Training checks: gradient accumulation, evaluation, checkpoints, resume, mixed precision."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models.gpt import Oglodon
from ..training.trainer import Trainer
from .registry import Ctx, Outcome, Unsupported, register
from .report import Status


def _require_fp32(ctx: Ctx) -> None:
    if ctx.dtype not in (torch.float32, torch.float64):
        raise Unsupported(f"training checks run in float32/float64, not {ctx.dtype_name}")


def _samples(ctx: Ctx, n: int, T: int, ragged: bool = True) -> list[dict[str, torch.Tensor]]:
    """``n`` random rows; with ``ragged`` the number of supervised tokens differs from row to row."""
    ids = ctx.ids(n, T).cpu()
    out = []
    for i in range(n):
        labels = ids[i].clone()
        if ragged:
            labels[: (i * 3) % (T - 2)] = -100
        out.append({"input_ids": ids[i], "labels": labels})
    return out


def _collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}


def _trainer(model, rows, batch_size, **kw) -> Trainer:
    dl = DataLoader(rows, batch_size=batch_size, shuffle=False)
    kw.setdefault("logger", None)
    kw.setdefault("log_interval", 0)
    kw.setdefault("device", next(model.parameters()).device)
    return Trainer(model, dl, **kw)


def _grads(model) -> torch.Tensor:
    return torch.cat([p.grad.detach().double().flatten() for p in model.parameters()])


def _manual_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum-over-tokens cross-entropy / number of supervised tokens, written out explicitly."""
    lg, lb = logits[:, :-1].float(), labels[:, 1:]
    logp = lg - torch.logsumexp(lg, dim=-1, keepdim=True)
    keep = lb != -100
    picked = logp.gather(-1, lb.clamp(min=0)[..., None]).squeeze(-1)
    return -(picked * keep).sum() / keep.sum()


@register("training", "training.gradient_accumulation")
def gradient_accumulation(ctx: Ctx) -> Outcome:
    _require_fp32(ctx)
    T = min(12, ctx.cfg.max_seq_len)
    rows = _samples(ctx, 8, T)
    micro = [_collate(rows[i : i + 2]) for i in range(0, 8, 2)]
    micro = [{k: v.to(ctx.device) for k, v in b.items()} for b in micro]
    union = {k: torch.cat([b[k] for b in micro]) for k in micro[0]}

    m1 = ctx.fresh()
    m1.train()
    _trainer(m1, rows, 2, grad_accum_steps=4).compute_gradients(micro)
    accumulated = _grads(m1)

    m2 = ctx.fresh()  # reference: ONE forward over the whole batch, loss normalised by total tokens
    _manual_loss(m2(union["input_ids"]), union["labels"]).backward()
    reference = _grads(m2)

    m3 = ctx.fresh()  # the tempting-but-wrong alternative: average of per-micro-batch means
    for b in micro:
        (_manual_loss(m3(b["input_ids"]), b["labels"]) / len(micro)).backward()
    naive = _grads(m3)

    scale = max(1.0, reference.abs().max().item())
    tol = ctx.tol["equiv"] * (1 + scale)
    diff = (accumulated - reference).abs().max().item()
    out = ctx.numeric(
        diff, tol, "accumulated micro-batch gradients == one large batch (token-weighted normalisation)"
    )
    naive_diff = (naive - reference).abs().max().item()
    if out.status is Status.PASS and naive_diff <= 10 * tol:
        return Outcome(
            Status.WARNING,
            f"equivalence holds but the test is not sensitive (naive mean-of-means differs by only {naive_diff:.1e})",
            diff,
            tol,
        )
    out.detail += f" (the naive mean of micro-batch means would differ by {naive_diff:.1e})"
    return out


@register("training", "training.eval_loss_weighting")
def eval_loss_weighting(ctx: Ctx) -> Outcome:
    _require_fp32(ctx)
    T = min(12, ctx.cfg.max_seq_len)
    rows = _samples(ctx, 6, T)
    m = ctx.fresh()
    tr = _trainer(m, rows, 2, eval_dataloader=DataLoader(rows, batch_size=2, shuffle=False))  # type: ignore[arg-type]
    got = tr.evaluate()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, 6, 2):
            b = {k: v.to(ctx.device) for k, v in _collate(rows[i : i + 2]).items()}
            lg, lb = m(b["input_ids"])[:, :-1].float(), b["labels"][:, 1:]
            logp = lg - torch.logsumexp(lg, dim=-1, keepdim=True)
            keep = lb != -100
            total += float(-(logp.gather(-1, lb.clamp(min=0)[..., None]).squeeze(-1) * keep).sum())
            count += int(keep.sum())
    return ctx.numeric(
        abs(got - total / count),
        ctx.tol["equiv"],
        "Trainer.evaluate() == token-weighted mean loss over batches with unequal token counts",
    )


@register("training", "training.checkpoint_roundtrip")
def checkpoint_roundtrip(ctx: Ctx) -> Outcome:
    problems = []
    model = ctx.fresh(stress=False)
    x = ctx.ids(2, min(10, ctx.cfg.max_seq_len))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt"
        tok = ctx.tokenizer if hasattr(ctx.tokenizer, "save") else None
        model.save_pretrained(path, tokenizer=tok)
        for f in ("config.json", "model.safetensors"):
            if not (path / f).exists():
                problems.append(f"{f} was not written")
        loaded = Oglodon.from_pretrained(path, device=ctx.device).to(ctx.dtype).eval()
    if loaded.config != model.config:
        problems.append("config changed in the round trip")
    a, b = model.state_dict(), loaded.state_dict()
    bad = [k for k in a if k not in b or not torch.equal(a[k], b[k])]
    if bad or set(a) != set(b):
        problems.append(f"weights differ after reload: {bad[:3]}")
    if model.config.tie_embeddings and loaded.lm_head.weight is not loaded.embed_tokens.weight:
        problems.append("tied embedding is no longer shared after reload")
    with torch.no_grad():
        diff = (model(x) - loaded(x)).abs().max().item()
        same_gen = torch.equal(
            model.generate(x[:, :3], max_new_tokens=4, eos_token_id=None),
            loaded.generate(x[:, :3], max_new_tokens=4, eos_token_id=None),
        )
    if not same_gen:
        problems.append("generation differs after reload")
    if problems:
        return Outcome(Status.FAIL, "; ".join(problems), diff, 0.0)
    return ctx.numeric(
        diff,
        0.0,
        "save_pretrained -> from_pretrained gives identical weights, logits and generation (safetensors)",
    )


@register("training", "training.resume_equivalence")
def resume_equivalence(ctx: Ctx) -> Outcome:
    _require_fp32(ctx)
    T = min(10, ctx.cfg.max_seq_len)
    rows = _samples(ctx, 16, T)
    base = ctx.fresh()  # keeps whatever dropout the *given* model has: exercises RNG restoration
    for src, dst in zip(ctx.model.modules(), base.modules(), strict=True):
        if isinstance(src, torch.nn.Dropout):
            dst.p = src.p
    kw = dict(max_steps=4, grad_accum_steps=2, learning_rate=1e-3, warmup_steps=1)

    def run(until=None, resume_from=None, save_to=None):
        torch.manual_seed(ctx.seed)
        m = copy.deepcopy(base)
        tr = _trainer(m, rows, 2, **kw)
        if resume_from:
            tr.load_checkpoint(resume_from)
        tr.train(until=until)
        if save_to:
            tr.save_checkpoint(save_to)
        return torch.cat([p.detach().double().flatten() for p in m.parameters()])

    with tempfile.TemporaryDirectory() as tmp:
        straight = run()
        run(until=2, save_to=tmp)
        resumed = run(resume_from=tmp)
    cpu = ctx.device.type == "cpu"
    return ctx.numeric(
        (straight - resumed).abs().max().item(),
        0.0 if cpu else 1e-6,
        "2 steps + save + load + 2 steps == 4 uninterrupted steps (weights, optimizer, schedule, data position, RNG)",
    )


@register("training", "training.mixed_precision_bf16")
def mixed_precision_bf16(ctx: Ctx) -> Outcome:
    _require_fp32(ctx)
    if ctx.device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise Unsupported("this GPU does not support bfloat16")
    T = min(12, ctx.cfg.max_seq_len)
    rows = _samples(ctx, 4, T, ragged=False)
    batch = [{k: v.to(ctx.device) for k, v in _collate(rows).items()}]
    losses, grads = [], []
    for prec in ("fp32", "bf16"):
        m = ctx.fresh()
        m.train()
        try:
            tr = _trainer(m, rows, 4, precision=prec)
            losses.append(tr.compute_gradients(batch))
        except RuntimeError as e:  # e.g. bf16 autocast kernels missing on this CPU
            raise Unsupported(f"bfloat16 autocast is not usable here: {e}") from e
        grads.append(_grads(m))
    if not torch.isfinite(grads[1]).all():
        return Outcome(Status.FAIL, "non-finite gradients under bf16 autocast", dtype="bfloat16 (autocast)")
    scale = max(1.0, abs(losses[0]))
    out = ctx.numeric(
        abs(losses[0] - losses[1]),
        5e-2 * scale,
        f"bf16-autocast loss {losses[1]:.4f} vs fp32 {losses[0]:.4f}; gradients finite",
    )
    out.dtype = f"bfloat16 (autocast on {ctx.device.type})"
    return out


@register("training", "training.mixed_precision_fp16")
def mixed_precision_fp16(ctx: Ctx) -> Outcome:
    _require_fp32(ctx)
    if ctx.device.type != "cuda":
        raise Unsupported(
            "fp16 training uses CUDA autocast + GradScaler; no CUDA device here, so this was NOT executed"
        )
    rows = _samples(ctx, 8, min(12, ctx.cfg.max_seq_len))  # pragma: no cover - needs a GPU
    m = ctx.fresh()  # pragma: no cover
    tr = _trainer(m, rows, 2, precision="fp16", max_steps=3, learning_rate=1e-3)  # pragma: no cover
    hist = tr.train()  # pragma: no cover
    ok = (
        all(torch.isfinite(p).all() for p in m.parameters())
        and tr.scaler is not None
        and tr.scaler.get_scale() > 0
    )  # pragma: no cover
    scale = tr.scaler.get_scale() if tr.scaler else float("nan")  # pragma: no cover
    out = Outcome(
        Status.PASS if ok else Status.FAIL,
        f"3 fp16 steps; final grad-scale {scale:.0f}; history {len(hist)}",
    )  # pragma: no cover
    out.dtype = "float16 (autocast on cuda)"  # pragma: no cover
    return out  # pragma: no cover
