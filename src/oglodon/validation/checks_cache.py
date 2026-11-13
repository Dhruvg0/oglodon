"""KV-cache checks. Cached paths are compared with the float64 reference and with uncached paths."""

from __future__ import annotations

import torch

from ..layers.cache import KVCache
from .reference import reference_forward, reference_model
from .registry import Ctx, Outcome, register
from .report import Status


def _stepwise_logits(model, ids: torch.Tensor, split: int) -> tuple[torch.Tensor, KVCache]:
    """Prefill ``ids[:, :split]`` then feed one token at a time through a KV cache."""
    out = model(ids[:, :split], use_cache=True)
    pieces, cache = [out.logits], out.past_key_values
    for i in range(split, ids.shape[1]):
        pieces.append(model(ids[:, i : i + 1], past_key_values=cache, use_cache=True).logits)
    return torch.cat(pieces, dim=1), cache


def _seq(ctx: Ctx, cap: int = 14) -> int:
    return min(cap, ctx.cfg.max_seq_len)


@register("kv_cache", "kv_cache.logits_vs_reference")
def logits_vs_reference(ctx: Ctx) -> Outcome:
    T = _seq(ctx)
    if T < 3:
        return Outcome(Status.SKIPPED, "max_seq_len too small")
    ids = ctx.ids(2, T)
    with torch.no_grad():
        cached, cache = _stepwise_logits(ctx.stress, ids, split=max(1, T // 3))
        ref = reference_forward(reference_model(ctx.stress), ids)
    d = (cached.double().cpu() - ref).abs().max().item()
    if cache.seq_len != T:
        return ctx.exact(
            False, f"cache holds {cache.seq_len} positions after feeding {T} tokens", abs(cache.seq_len - T)
        )
    return ctx.numeric(
        d,
        ctx.tol["ref"],
        f"prefill + {T - max(1, T // 3)} single-token cached steps == float64 reference over the full sequence",
        ref.abs().max().item(),
    )


@register("kv_cache", "kv_cache.generation_equivalence")
def generation_equivalence(ctx: Ctx) -> Outcome:
    cfg, m = ctx.cfg, ctx.stress
    T0 = min(5, cfg.max_seq_len - 1)
    new = min(10, cfg.max_seq_len - T0)
    if new < 2:
        return Outcome(Status.SKIPPED, "max_seq_len too small to generate several tokens")
    prompt = ctx.ids(3, T0)
    greedy = {
        c: m.generate(prompt, max_new_tokens=new, eos_token_id=None, use_cache=c) for c in (True, False)
    }
    sampled = {
        c: m.generate(prompt, max_new_tokens=new, eos_token_id=None, use_cache=c, temperature=1.0, top_k=8,
                      generator=torch.Generator().manual_seed(ctx.seed))
        for c in (True, False)
    }  # fmt: skip
    mism = int((greedy[True] != greedy[False]).sum() + (sampled[True] != sampled[False]).sum())
    if mism:
        return ctx.exact(False, f"cached and uncached generation differ in {mism} tokens", mism)
    # the generated tokens must also be what the *reference* model would pick (teacher forcing)
    with torch.no_grad():
        ref = reference_forward(reference_model(m), greedy[True])
    top2 = ref.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    pred = ref.argmax(-1)
    wrong = 0
    for t in range(T0, greedy[True].shape[1]):
        decisive = margin[:, t - 1] > 1e-6
        wrong += int(((pred[:, t - 1] != greedy[True][:, t].cpu()) & decisive).sum())
    return ctx.exact(
        wrong == 0,
        f"greedy and seeded-sampling output identical with/without cache; {wrong} tokens differ from the reference argmax",
        wrong,
    )


@register("kv_cache", "kv_cache.positions")
def positions(ctx: Ctx) -> Outcome:
    cfg, m = ctx.cfg, ctx.stress
    T = _seq(ctx, 10)
    ids = ctx.ids(2, T)
    worst = 0.0
    with torch.no_grad():
        # (a) explicit position ids in cached steps == implicit ones
        implicit, _ = _stepwise_logits(m, ids, T // 2)
        scale = implicit.abs().max().item()
        out = m(
            ids[:, : T // 2],
            position_ids=torch.arange(T // 2, device=ctx.device).expand(2, -1),
            use_cache=True,
        )
        pieces, cache = [out.logits], out.past_key_values
        for i in range(T // 2, T):
            pos = torch.full((2, 1), i, device=ctx.device)
            pieces.append(
                m(ids[:, i : i + 1], position_ids=pos, past_key_values=cache, use_cache=True).logits
            )
        worst = max(worst, (implicit - torch.cat(pieces, 1)).abs().max().item())
        # (b) a left-padded batch generates, with the cache, exactly what each row generates alone
    new = min(6, cfg.max_seq_len - T)
    if new >= 1:
        pads = [0, 3]
        rows = [ctx.ids(1, T - p) for p in pads]
        batch = torch.zeros(2, T, dtype=torch.long, device=ctx.device)
        for b, (p, r) in enumerate(zip(pads, rows, strict=True)):
            batch[b, p:] = r[0]
        mask = _left_pad_mask(2, T, pads, ctx.device)
        got = m.generate(batch, attention_mask=mask, max_new_tokens=new, eos_token_id=None)
        for b, (p, r) in enumerate(zip(pads, rows, strict=True)):
            solo = m.generate(r, max_new_tokens=new, eos_token_id=None)
            if not torch.equal(got[b, T:], solo[0, T - p :]):
                return ctx.exact(
                    False,
                    f"left-padded row {b} (pad {p}) generated different tokens than the same prompt alone",
                    1,
                )
    return ctx.numeric(
        worst,
        ctx.tol["equiv"],
        "explicit == implicit positions in cached decoding; left-padded cached generation == unpadded",
        scale,
    )


def _left_pad_mask(batch, seq, pads, device):
    m = torch.ones(batch, seq, dtype=torch.long)
    for b, p in enumerate(pads):
        m[b, :p] = 0
    return m.to(device)


@register("kv_cache", "kv_cache.multi_step_and_chunking")
def multi_step(ctx: Ctx) -> Outcome:
    cfg, m = ctx.cfg, ctx.stress
    T = _seq(ctx, 12)
    if T < 4:
        return Outcome(Status.SKIPPED, "max_seq_len too small")
    ids = ctx.ids(2, T)
    worst = 0.0
    el = torch.empty((), dtype=ctx.dtype).element_size()
    with torch.no_grad():
        full = m(ids)
        plans = ([1] * T, [3, 4, T - 7], [T - 1, 1], [T])
        for sizes in (p for p in plans if min(p) > 0 and sum(p) == T):
            cache, fed, pieces = KVCache(cfg.num_layers), 0, []
            for s in sizes:
                pieces.append(m(ids[:, fed : fed + s], past_key_values=cache, use_cache=True).logits)
                fed += s
                if cache.seq_len != fed:
                    return ctx.exact(
                        False, f"cache length {cache.seq_len} != tokens fed {fed} (chunks {sizes})", 1
                    )
            worst = max(worst, (torch.cat(pieces, 1) - full).abs().max().item())
            expect = 2 * cfg.num_layers * 2 * cfg.num_kv_heads * T * cfg.head_dim * el  # K+V, layers, B=2
            if cache.nbytes() != expect:
                return ctx.exact(
                    False,
                    f"cache holds {cache.nbytes()} bytes, expected {expect} from the config",
                    abs(cache.nbytes() - expect),
                )
            key0 = cache.keys[0]
            if key0 is None or tuple(key0.shape) != (2, cfg.num_kv_heads, T, cfg.head_dim):
                return ctx.exact(
                    False, f"unexpected key cache shape {None if key0 is None else tuple(key0.shape)}", 1
                )
    return ctx.numeric(
        worst,
        ctx.tol["equiv"],
        "token-by-token and chunked prefill == full forward; cache length, shape and byte size match the config",
        full.abs().max().item(),
    )


@register("kv_cache", "kv_cache.reset_and_batch")
def reset_and_batch(ctx: Ctx) -> Outcome:
    m = ctx.stress
    T = _seq(ctx, 8)
    ids = ctx.ids(3, T)
    worst = 0.0
    with torch.no_grad():
        first = m(ids, use_cache=True)
        cache = first.past_key_values
        cache.reset()
        if cache.seq_len != 0 or any(k is not None for k in cache.keys):
            return ctx.exact(False, "reset() left data in the cache", 1)
        again = m(ids, past_key_values=cache, use_cache=True)
        worst = max(worst, (first.logits - again.logits).abs().max().item())
        # reusing a cache with a different batch size must fail loudly, not corrupt silently
        try:
            m(ctx.ids(2, 1), past_key_values=again.past_key_values, use_cache=True)
        except ValueError:
            pass
        else:
            return ctx.exact(False, "a cache built for batch 3 silently accepted a batch of 2", 1)
        # rows are independent: row 1 of a batch == that row alone
        solo = m(ids[1:2], use_cache=True).logits
        worst = max(worst, (first.logits[1:2] - solo).abs().max().item())
        # generate() keeps no state between calls and restores training mode
        m.train()
        a = m.generate(ids[:, :3], max_new_tokens=3, eos_token_id=None)
        restored = m.training
        m.eval()
        b = m.generate(ids[:, :3], max_new_tokens=3, eos_token_id=None)
    if not restored:
        return ctx.exact(False, "generate() did not restore training mode", 1)
    if not torch.equal(a, b):
        return ctx.exact(False, "two identical generate() calls returned different tokens (state leaked)", 1)
    return ctx.numeric(
        worst,
        ctx.tol["equiv"],
        "reset() empties the cache; batch-size change raises; rows independent; generate() stateless",
        first.logits.abs().max().item(),
    )
