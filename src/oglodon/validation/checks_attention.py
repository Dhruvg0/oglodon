"""Attention checks: reference agreement, eager vs SDPA, masks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..layers.attention import CausalSelfAttention, build_attention_bias
from .reference import ref_attention, ref_mask, reference_forward, reference_model
from .registry import Ctx, Outcome, Unsupported, register
from .report import Status


def _d64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().double().cpu()


def _left_pad_mask(batch: int, seq: int, pads: list[int], device) -> torch.Tensor:
    m = torch.ones(batch, seq, dtype=torch.long)
    for b, p in enumerate(pads[:batch]):
        m[b, :p] = 0
    return m.to(device)


def _positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask.cumsum(-1) - 1).clamp(min=0)


def _attn_vs_reference(ctx: Ctx, module: CausalSelfAttention, rope_module, scenarios) -> float:
    """Max |module - reference| over the non-padding query rows of every scenario."""
    cfg, worst = ctx.cfg, 0.0
    B, T = 2, min(12, cfg.max_seq_len)
    x = ctx.randn(B, T, cfg.hidden_size)
    pos = torch.arange(T, device=ctx.device).expand(B, T)
    rope = rope_module(pos) if rope_module is not None else None
    for pad, window in scenarios:
        mask = _left_pad_mask(B, T, [pad, 0], ctx.device) if pad else None
        old = module.sliding_window
        module.sliding_window = window
        try:
            got, _ = module(x, rope=rope, key_padding_mask=mask)
        finally:
            module.sliding_window = old
        want = ref_attention(
            _d64(x), _d64(module.q_proj.weight), _d64(module.k_proj.weight), _d64(module.v_proj.weight),
            _d64(module.o_proj.weight), module.num_heads, module.num_kv_heads,
            pos.cpu() if rope_module is not None else None, cfg.rope_base,
            None if mask is None else mask.cpu(), window,
            bq=None if module.q_proj.bias is None else _d64(module.q_proj.bias),
            bk=None if module.k_proj.bias is None else _d64(module.k_proj.bias),
            bv=None if module.v_proj.bias is None else _d64(module.v_proj.bias),
            alibi=module.alibi,
        )  # fmt: skip
        keep = torch.ones(B, T, dtype=torch.bool) if mask is None else mask.cpu().bool()
        worst = max(worst, (_d64(got) - want).abs()[keep].max().item())
    return worst


@register("attention", "attention.matches_reference")
def attention_matches_reference(ctx: Ctx) -> Outcome:
    block = ctx.stress.blocks[0].attn
    scen = [(0, None), (3, None), (0, 4), (4, 5)]
    d = _attn_vs_reference(ctx, block, getattr(ctx.stress, "rope", None), scen)
    return ctx.numeric(
        d,
        ctx.tol["ref"],
        f"{ctx.cfg.attention_type} attention vs float64 reference (causal, left padding, local window)",
    )


@register("attention", "attention.layouts_vs_reference")
def attention_layouts(ctx: Ctx) -> Outcome:
    cfg, worst, names = ctx.cfg, 0.0, []
    for kv in [k for k in range(1, cfg.num_heads + 1) if cfg.num_heads % k == 0]:
        m = CausalSelfAttention(cfg.hidden_size, cfg.num_heads, kv).to(ctx.device, ctx.dtype)
        g = torch.Generator().manual_seed(ctx.seed + kv)
        with torch.no_grad():
            for p in m.parameters():
                p.copy_((torch.randn(p.shape, generator=g) * 0.3).to(ctx.device, ctx.dtype))
        worst = max(worst, _attn_vs_reference(ctx, m, getattr(ctx.stress, "rope", None), [(3, None)]))
        names.append("MHA" if kv == cfg.num_heads else "MQA" if kv == 1 else f"GQA(kv={kv})")
    return ctx.numeric(worst, ctx.tol["ref"], "all layouts vs reference: " + ", ".join(names))


def _grad_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.detach().double().flatten() for p in model.parameters() if p.grad is not None])


@register("attention", "attention.eager_vs_sdpa")
def eager_vs_sdpa(ctx: Ctx) -> Outcome:
    if not hasattr(F, "scaled_dot_product_attention"):
        raise Unsupported("this torch build has no scaled_dot_product_attention")
    cfg = ctx.cfg
    B, T = 3, min(12, cfg.max_seq_len)
    worst_out, worst_grad, scale = 0.0, 0.0, 1.0
    for pads, window in (([0, 0, 0], None), ([0, 3, 5], None), ([0, 3, 5], 4)):
        mask = _left_pad_mask(B, T, pads, ctx.device)
        pos, ids = _positions_from_mask(mask), ctx.ids(B, T)
        labels = ids.masked_fill(mask == 0, -100)
        # a padded *query* row has nothing to attend to; exclude the predictions made from it
        labels[:, 1:] = labels[:, 1:].masked_fill(mask[:, :-1] == 0, -100)
        outs, grads = [], []
        for impl in ("eager", "sdpa"):
            m = ctx.fresh()
            m.set_attn_impl(impl)
            for blk in m.blocks:
                blk.attn.sliding_window = window
            out = m(ids, attention_mask=mask, position_ids=pos, labels=labels)
            out.loss.backward()
            outs.append(out.logits.detach())
            grads.append(_grad_vector(m))
        keep = mask.bool()
        if not all(torch.isfinite(o).all() for o in outs):
            return Outcome(Status.FAIL, "non-finite logits in eager or sdpa path")
        worst_out = max(worst_out, (outs[0] - outs[1]).double().abs()[keep].max().item())
        worst_grad = max(worst_grad, (grads[0] - grads[1]).abs().max().item())
        scale = max(scale, grads[0].abs().max().item(), outs[0].abs().max().item())
    tol_g = ctx.tol["equiv"] * (1 + scale)
    diff, tol = max(worst_out, worst_grad), max(ctx.tol["equiv"], tol_g)
    return ctx.numeric(
        diff,
        tol,
        f"eager == sdpa for logits ({worst_out:.1e}) and gradients ({worst_grad:.1e}); causal, padded, windowed",
    )


@register("attention", "attention.causal_mask")
def causal_mask(ctx: Ctx) -> Outcome:
    m, cfg = ctx.stress, ctx.cfg
    B, T = 2, min(16, cfg.max_seq_len)
    t = T // 2
    x = ctx.ids(B, T)
    y = x.clone()
    y[:, t + 1 :] = (y[:, t + 1 :] + 1 + ctx.ids(B, T - t - 1)) % cfg.vocab_size
    with torch.no_grad():
        a, b = m(x), m(y)
        weights = m(x, output_attentions=True).attentions
    past = (a[:, : t + 1] - b[:, : t + 1]).abs().max().item()
    future = (a[:, t + 1 :] - b[:, t + 1 :]).abs().max().item()
    above = max(w.triu(1).abs().max().item() for w in weights)
    if future <= ctx.tol["equiv"]:
        return Outcome(
            Status.FAIL,
            "changing future tokens never changed the later logits: the test cannot detect leakage",
        )
    out = ctx.numeric(
        max(past, above),
        0.0 if past == 0 and above == 0 else ctx.tol["equiv"],
        f"future tokens do not change past logits; attention weights above the diagonal are zero (future effect {future:.1e})",
    )
    return out


@register("attention", "attention.no_future_gradient")
def no_future_gradient(ctx: Ctx) -> Outcome:
    m = ctx.fresh()
    B, T = 2, min(12, ctx.cfg.max_seq_len)
    t = T // 2
    store = {}

    def hook(_m, _i, out):
        out.retain_grad()
        store["emb"] = out

    h = m.embed_tokens.register_forward_hook(hook)
    try:
        m(ctx.ids(B, T))[:, t].sum().backward()
    finally:
        h.remove()
    g = store["emb"].grad
    future, visible = g[:, t + 1 :].abs().max().item(), g[:, : t + 1].abs().max().item()
    if visible == 0:
        return Outcome(Status.FAIL, "no gradient reached visible tokens: test is vacuous")
    return ctx.numeric(
        future,
        0.0,
        f"d logits[t] / d embeddings of later tokens is exactly zero (visible-token gradient {visible:.1e})",
    )


@register("attention", "attention.padding_masks")
def padding_masks(ctx: Ctx) -> Outcome:
    cfg = ctx.cfg
    T = min(10, cfg.max_seq_len)
    pads = [0, 3, 5]
    B = len(pads)
    base = ctx.ids(B, T)
    mask = _left_pad_mask(B, T, pads, ctx.device)
    pos = _positions_from_mask(mask)
    m = ctx.stress
    with torch.no_grad():
        got = m(base, attention_mask=mask, position_ids=pos)
        worst = 0.0
        for b, p in enumerate(pads):  # (a) left padding == the unpadded sequence on its own
            solo = m(base[b : b + 1, p:])
            worst = max(worst, (got[b, p:] - solo[0]).abs().max().item())
        rm = reference_model(m)  # (b) ... and == the independent reference with the same mask
        ref = reference_forward(rm, base, mask, pos)
        keep = mask.cpu().bool()
        worst = max(worst, (got.double().cpu() - ref.detach()).abs()[keep].max().item())
        # (c) right padding: content of padded slots must not influence real positions
        rmask = torch.ones(B, T, dtype=torch.long, device=ctx.device)
        for b, p in enumerate(pads):
            rmask[b, T - p :] = 0 if p else 1
        alt = base.clone()
        alt[rmask == 0] = (alt[rmask == 0] + 1) % cfg.vocab_size
        r1 = m(base, attention_mask=rmask)
        r2 = m(alt, attention_mask=rmask)
        worst = max(worst, (r1 - r2).abs()[rmask.bool()].max().item())
        finite = bool(torch.isfinite(got).all() and torch.isfinite(r1).all())  # (d) padded rows stay finite
    if not finite:
        return Outcome(Status.FAIL, "non-finite logits at padded positions")
    return ctx.numeric(
        worst,
        ctx.tol["ref"],
        "left padding == unpadded == reference; right-pad content is ignored; padded rows finite",
        got.abs().max().item(),
    )


@register("attention", "attention.rope_relative_shift")
def rope_relative_shift(ctx: Ctx) -> Outcome:
    if not ctx.cfg.use_rope:
        raise Unsupported("model uses learned absolute positions, not RoPE (no shift invariance expected)")
    T = min(12, ctx.cfg.max_seq_len - 5)
    if T < 2:
        raise Unsupported("max_seq_len too small to test a position shift")
    x = ctx.ids(2, T)
    with torch.no_grad():
        a = ctx.stress(x)
        b = ctx.stress(x, position_ids=(torch.arange(T, device=ctx.device) + 5).expand(2, T))
    return ctx.numeric(
        (a - b).abs().max().item(),
        ctx.tol["ref"],
        "RoPE: shifting all positions by 5 leaves logits unchanged (relative encoding)",
        a.abs().max().item(),
    )


@register("attention", "attention.local_mask")
def local_mask(ctx: Ctx) -> Outcome:
    # (1) the mask builder against an entry-by-entry specification
    g = torch.Generator().manual_seed(ctx.seed)
    mismatches = 0
    for q_len, kv_len in ((6, 6), (1, 9), (3, 8), (4, 4)):
        for window in (None, 1, 3, 5):
            for use_pad in (False, True):
                pad = (torch.rand(2, kv_len, generator=g) > 0.3).long() if use_pad else None
                bias = build_attention_bias(q_len, kv_len, torch.device("cpu"), torch.float32, pad, window)
                got = bias == 0
                want = ref_mask(q_len, kv_len, pad, window)
                mismatches += int((got != want[:, None].expand_as(got)).sum())
    if mismatches:
        return ctx.exact(
            False, f"mask builder disagrees with the specification in {mismatches} entries", mismatches
        )
    # (2) the whole model with a window forced on, against the float64 reference
    w = 3
    m = ctx.fresh()
    for blk in m.blocks:
        blk.attn.sliding_window = w
    T = min(10, ctx.cfg.max_seq_len)
    x = ctx.ids(2, T)
    with torch.no_grad():
        got = m(x)
        ref = reference_forward(reference_model(m), x, window=w)
        plain = ctx.fresh()
        for blk in plain.blocks:
            blk.attn.sliding_window = None
        unwindowed = plain(x)
    if (got - unwindowed).abs().max().item() < 10 * ctx.tol["equiv"] and w < T:
        return Outcome(Status.FAIL, "enabling a window did not change the output: the window is not applied")
    return ctx.numeric(
        (got.double().cpu() - ref.detach()).abs().max().item(),
        ctx.tol["ref"],
        f"mask builder == specification ({mismatches} mismatches); model with window={w} == reference",
        ref.abs().max().item(),
    )
