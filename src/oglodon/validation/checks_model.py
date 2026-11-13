"""Model-level checks: shapes, finiteness, configuration consistency, reference agreement."""

from __future__ import annotations

import torch

from ..config import OglodonConfig
from .reference import reference_forward, reference_loss, reference_model
from .registry import Ctx, Outcome, register
from .report import Status


def expected_num_parameters(cfg: OglodonConfig) -> int:
    """Parameter count derived from the config alone (does not look at the modules)."""
    d, h, hd = cfg.hidden_size, cfg.intermediate_size, cfg.head_dim
    norm = d if cfg.use_rmsnorm else 2 * d  # RMSNorm: gain; LayerNorm: gain + bias
    attn = d * cfg.num_heads * hd + 2 * d * cfg.num_kv_heads * hd + cfg.num_heads * hd * d
    if cfg.qkv_bias:
        attn += cfg.num_heads * hd + 2 * cfg.num_kv_heads * hd
    ffn = (3 if cfg.use_swiglu else 2) * d * h  # the activation has no parameters
    total = cfg.vocab_size * d + cfg.num_layers * (2 * norm + attn + ffn) + norm
    if not cfg.use_rope and not cfg.use_alibi:
        total += cfg.max_seq_len * d
    if not cfg.tie_embeddings:
        total += cfg.vocab_size * d
    return total


@register("model", "model.config_matches")
def config_matches(ctx: Ctx) -> Outcome:
    want = ctx.requested_config
    if want is None or want.to_dict() == ctx.cfg.to_dict():
        OglodonConfig.from_dict(ctx.cfg.to_dict())  # re-runs every validation rule
        return Outcome(
            Status.PASS,
            f"{ctx.cfg.attention_type}, head_dim={ctx.cfg.head_dim}, config valid and equal to the requested one",
        )
    diff = {k: (want.to_dict()[k], v) for k, v in ctx.cfg.to_dict().items() if want.to_dict()[k] != v}
    return Outcome(Status.FAIL, f"model.config differs from the given config (given, actual): {diff}")


@register("model", "model.parameters_match_config")
def parameters_match_config(ctx: Ctx) -> Outcome:
    cfg, m = ctx.cfg, ctx.model
    problems = []
    got, want = m.num_parameters(), expected_num_parameters(cfg)
    if got != want:
        problems.append(f"parameter count {got} != {want} implied by the config")
    if len(m.blocks) != cfg.num_layers:
        problems.append(f"{len(m.blocks)} blocks != num_layers={cfg.num_layers}")
    a = m.blocks[0].attn
    shapes = {
        "q_proj": (cfg.num_heads * cfg.head_dim, cfg.hidden_size),
        "k_proj": (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
        "v_proj": (cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size),
        "o_proj": (cfg.hidden_size, cfg.num_heads * cfg.head_dim),
    }
    for n, s in shapes.items():
        if tuple(getattr(a, n).weight.shape) != s:
            problems.append(f"{n} has shape {tuple(getattr(a, n).weight.shape)}, expected {s}")
    if (m.lm_head.weight is m.embed_tokens.weight) != cfg.tie_embeddings:
        problems.append(
            f"tie_embeddings={cfg.tie_embeddings} but weights are {'shared' if cfg.tie_embeddings is False else 'not shared'}"
        )
    if (m.rope is not None) != cfg.use_rope:
        problems.append("use_rope does not match the modules")
    return Outcome(
        Status.FAIL if problems else Status.PASS,
        "; ".join(problems)
        or f"{got:,} parameters == config-derived count; projection shapes and weight tying match",
    )


@register("model", "model.vocab_and_embeddings")
def vocab_and_embeddings(ctx: Ctx) -> Outcome:
    cfg, m = ctx.cfg, ctx.model
    problems = []
    if m.embed_tokens.weight.shape != (cfg.vocab_size, cfg.hidden_size):
        problems.append(f"embedding {tuple(m.embed_tokens.weight.shape)} != (vocab_size, hidden_size)")
    if m.lm_head.weight.shape[0] != cfg.vocab_size:
        problems.append("lm_head output size != vocab_size")
    for n in ("pad_token_id", "eos_token_id"):
        v = getattr(cfg, n)
        if v is not None and not 0 <= v < cfg.vocab_size:
            problems.append(f"{n}={v} outside the vocabulary")
    with torch.no_grad():
        if m(ctx.ids(1, 3)).shape[-1] != cfg.vocab_size:
            problems.append("logits width != vocab_size")
    tok = ctx.tokenizer
    if tok is None:
        if problems:
            return Outcome(Status.FAIL, "; ".join(problems))
        return Outcome(
            Status.SKIPPED,
            "embedding/vocabulary shapes are fine, but no tokenizer was given so tokenizer compatibility was not checked (pass tokenizer=...)",
        )
    size = getattr(tok, "vocab_size", None)
    if size is None:
        return Outcome(Status.UNSUPPORTED, "tokenizer has no vocab_size attribute")
    if size > cfg.vocab_size:
        problems.append(
            f"tokenizer produces ids up to {size - 1} but the model has only {cfg.vocab_size} embeddings"
        )
    if problems:
        return Outcome(Status.FAIL, "; ".join(problems))
    warn = []
    for n in ("pad_token_id", "eos_token_id"):
        a, b = getattr(tok, n, None), getattr(cfg, n)
        if a is not None and b is not None and a != b:
            warn.append(f"{n}: tokenizer={a} config={b}")
    if size < cfg.vocab_size:
        warn.append(f"model has {cfg.vocab_size - size} unused embedding rows")
    if warn:
        return Outcome(Status.WARNING, "compatible, but: " + "; ".join(warn))
    return Outcome(Status.PASS, "tokenizer vocabulary fits the embedding table; special-token ids agree")


@register("model", "model.shape_invariants")
def shape_invariants(ctx: Ctx) -> Outcome:
    cfg, m = ctx.cfg, ctx.stress
    problems = []
    cases = {(1, 1), (2, min(7, cfg.max_seq_len)), (3, min(16, cfg.max_seq_len)), (1, cfg.max_seq_len)}
    with torch.no_grad():
        for B, T in sorted(cases):
            x = ctx.ids(B, T)
            out = m(x, labels=x, use_cache=True, output_attentions=T <= 64)
            if out.logits.shape != (B, T, cfg.vocab_size):
                problems.append(f"logits {tuple(out.logits.shape)} for input {(B, T)}")
            if out.loss is not None and out.loss.dim() != 0 and T > 1:
                problems.append("loss is not a scalar")
            if tuple(out.past_key_values.keys[-1].shape) != (B, cfg.num_kv_heads, T, cfg.head_dim):
                problems.append(f"cache shape {tuple(out.past_key_values.keys[-1].shape)} for input {(B, T)}")
            if out.attentions is not None and (
                len(out.attentions) != cfg.num_layers
                or tuple(out.attentions[0].shape) != (B, cfg.num_heads, T, T)
            ):
                problems.append(
                    f"attention maps have shape {tuple(out.attentions[0].shape)} for input {(B, T)}"
                )
        try:
            m(ctx.ids(1, cfg.max_seq_len + 1))
            problems.append("a sequence longer than max_seq_len was accepted")
        except ValueError:
            pass
    return Outcome(
        Status.FAIL if problems else Status.PASS,
        "; ".join(problems)
        or f"logits, loss, cache and attention-map shapes correct for {len(cases)} input shapes; over-long input rejected",
    )


@register("model", "model.finite_outputs_and_gradients")
def finite_outputs(ctx: Ctx) -> Outcome:
    cfg = ctx.cfg
    x = ctx.ids(2, min(12, cfg.max_seq_len))
    problems: list[str] = []
    warns: list[str] = []
    for label, model in (("given weights", ctx.fresh(stress=False)), ("stress weights", ctx.fresh())):
        model.zero_grad(set_to_none=True)
        out = model(x, labels=x)
        out.loss.backward()
        if not (torch.isfinite(out.logits).all() and torch.isfinite(out.loss)):
            problems.append(f"{label}: non-finite logits or loss")
        for n, p in model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                problems.append(f"{label}: gradient of {n} is missing or non-finite")
            elif p.grad.abs().sum() == 0:
                (problems if label == "stress weights" else warns).append(
                    f"{label}: gradient of {n} is exactly zero"
                )
    if problems:
        return Outcome(Status.FAIL, "; ".join(problems[:4]))
    if warns:
        return Outcome(Status.WARNING, "; ".join(warns[:4]))
    return Outcome(
        Status.PASS,
        "finite logits/loss; every parameter receives a finite non-zero gradient (given and stress weights)",
    )


@register("model", "model.reference_forward_backward")
def reference_forward_backward(ctx: Ctx) -> Outcome:
    cfg = ctx.cfg
    T = min(10, cfg.max_seq_len)
    x = ctx.ids(2, T)
    labels = x.clone()
    labels[0, : T // 3] = -100
    d_logits = d_loss = d_grad = 0.0
    scale = 1.0
    for model in (ctx.fresh(stress=False), ctx.fresh()):
        model.zero_grad(set_to_none=True)
        out = model(x, labels=labels)
        out.loss.backward()
        rm = reference_model(model)
        ref_logits = reference_forward(rm, x)
        ref_loss = reference_loss(ref_logits, labels)
        ref_loss.backward()
        d_logits = max(d_logits, (out.logits.double().cpu() - ref_logits.detach()).abs().max().item())
        d_loss = max(d_loss, abs(out.loss.item() - ref_loss.item()))
        for (n, p), (_, q) in zip(model.named_parameters(), rm.named_parameters(), strict=True):
            if q.grad is None:
                return Outcome(Status.FAIL, f"the reference computation does not use parameter {n}")
            d_grad = max(d_grad, (p.grad.double().cpu() - q.grad).abs().max().item())
            scale = max(scale, q.grad.abs().max().item())
    tol = ctx.tol["ref"]
    worst, limit = max(d_logits, d_loss, d_grad), max(tol, tol * (1 + scale))
    return ctx.numeric(
        worst,
        limit,
        f"logits ({d_logits:.1e}), loss ({d_loss:.1e}) and parameter gradients ({d_grad:.1e}) vs an independent float64 reference, given + stress weights",
    )


@register("model", "model.numerical_stability")
def numerical_stability(ctx: Ctx) -> Outcome:
    cfg, m = ctx.cfg, ctx.fresh()
    T = cfg.max_seq_len if cfg.max_seq_len <= 128 else 128
    x = ctx.ids(2, T)
    with torch.no_grad():
        mask = torch.ones_like(x)
        mask[0, : T // 2] = 0  # fully masked query rows
        padded = m(x, attention_mask=mask)
        longest = m(x)
        m.embed_tokens.weight.mul_(50.0)  # large activations
        big = m(x, output_attentions=True) if T <= 64 else m(x)
        big_logits = big.logits if T <= 64 else big
        sums = [w.float().sum(-1) for w in big.attentions] if T <= 64 else []
    bad = [
        n
        for n, t in (("padded", padded), ("longest", longest), ("50x embeddings", big_logits))
        if not torch.isfinite(t).all()
    ]
    if bad:
        return Outcome(Status.FAIL, f"non-finite logits: {', '.join(bad)}")
    dev = max((s - 1).abs().max().item() for s in sums) if sums else 0.0
    return ctx.numeric(
        dev,
        1e-3 if ctx.dtype in (torch.bfloat16, torch.float16) else 1e-5,
        f"finite logits for fully-masked rows, length {T} and 50x embeddings; attention rows sum to 1",
    )
