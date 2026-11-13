"""Independent float64 reference implementation of the Oglodon forward pass.

Deliberately written differently from ``oglodon.layers``: masks are decided per (query, key) pair by a
plain Python predicate, softmax is written out by hand over exactly the allowed keys, RoPE uses
complex numbers, and head sharing is an explicit ``h // group`` index. It only reads the weights of
the model under test; it shares no code with it. It is slow (python loops) and meant for small shapes.
"""

from __future__ import annotations

import copy
import math

import torch


def allowed(i_abs: int, j: int, key_ok: list[bool] | None, window: int | None) -> bool:
    """May a query at absolute position ``i_abs`` attend to key slot ``j``? (the specification)"""
    if j > i_abs:
        return False
    if window is not None and i_abs - j >= window:
        return False
    return key_ok is None or bool(key_ok[j])


def ref_mask(q_len: int, kv_len: int, key_padding: torch.Tensor | None, window: int | None) -> torch.Tensor:
    """Boolean ``(B, q_len, kv_len)`` mask built entry by entry from :func:`allowed`."""
    B = 1 if key_padding is None else key_padding.shape[0]
    out = torch.zeros(B, q_len, kv_len, dtype=torch.bool)
    for b in range(B):
        ok = None if key_padding is None else [bool(v) for v in key_padding[b].tolist()]
        for i in range(q_len):
            for j in range(kv_len):
                out[b, i, j] = allowed(kv_len - q_len + i, j, ok, window)
    return out


def ref_rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate pairs ``(x[2i], x[2i+1])`` by ``pos * base**(-2i/d)``; ``x``: ``(B,H,T,d)``, ``positions``: ``(B,T)``."""
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)  # (d/2,)
    z = torch.view_as_complex(x.reshape(*x.shape[:-1], d // 2, 2).contiguous())
    rot = torch.polar(torch.ones(1, dtype=torch.float64), positions[:, :, None].double() * theta)  # (B,T,d/2)
    return torch.view_as_real(z * rot[:, None]).reshape(x.shape)


def ref_attention(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    positions: torch.Tensor | None,
    rope_base: float,
    key_padding: torch.Tensor | None = None,
    window: int | None = None,
    bq: torch.Tensor | None = None,
    bk: torch.Tensor | None = None,
    bv: torch.Tensor | None = None,
    alibi: bool = False,
) -> torch.Tensor:
    """Causal self-attention over the whole sequence ``x`` ``(B,T,D)`` (no cache), float64."""
    B, T, _ = x.shape
    hd = wq.shape[0] // num_heads
    group = num_heads // num_kv_heads
    q = (x @ wq.T + (0 if bq is None else bq)).reshape(B, T, num_heads, hd).transpose(1, 2)
    k = (x @ wk.T + (0 if bk is None else bk)).reshape(B, T, num_kv_heads, hd).transpose(1, 2)
    v = (x @ wv.T + (0 if bv is None else bv)).reshape(B, T, num_kv_heads, hd).transpose(1, 2)
    if positions is not None:
        q, k = ref_rope(q, positions, rope_base), ref_rope(k, positions, rope_base)
    out = torch.zeros(B, num_heads, T, hd, dtype=x.dtype)
    rows = []
    for b in range(B):
        ok = None if key_padding is None else [bool(t) for t in key_padding[b].tolist()]
        per_head = []
        for h in range(num_heads):
            kv = h // group  # explicit head sharing
            per_query = []
            for i in range(T):
                idx = [j for j in range(T) if allowed(i, j, ok, window)]
                if not idx:  # a padding query with nothing to look at: the model uses a uniform row
                    idx = list(range(T))
                s = torch.stack([(q[b, h, i] * k[b, kv, j]).sum() for j in idx]) / math.sqrt(hd)
                if alibi:  # slope_h * (i - j); slopes written out from the paper's formula
                    s = s - _alibi_slope(h, num_heads) * torch.tensor(
                        [float(i - j) for j in idx], dtype=s.dtype
                    )
                e = torch.exp(s - s.max())
                p = e / e.sum()
                per_query.append(torch.stack([p[n] * v[b, kv, j] for n, j in enumerate(idx)]).sum(0))
            per_head.append(torch.stack(per_query))
        rows.append(torch.stack(per_head))
    out = torch.stack(rows)  # (B,H,T,hd)
    return out.transpose(1, 2).reshape(B, T, num_heads * hd) @ wo.T


def _alibi_slope(h: int, n: int) -> float:
    """Slope of head ``h`` of ``n`` (paper: 2^(-8(h+1)/n); non powers of two interleave two sequences)."""
    if n & (n - 1) == 0:
        return 2.0 ** (-8.0 * (h + 1) / n)
    m = 2 ** int(math.floor(math.log2(n)))
    return 2.0 ** (-8.0 * (h + 1) / m) if h < m else 2.0 ** (-4.0 * (2 * (h - m) + 1) / m)


def _norm(x: torch.Tensor, module, use_rmsnorm: bool, eps: float) -> torch.Tensor:
    if use_rmsnorm:
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * module.weight
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * module.weight + module.bias


def _act(h: torch.Tensor, name: str) -> torch.Tensor:
    if name == "silu":
        return h * torch.sigmoid(h)
    if name == "gelu":
        return 0.5 * h * (1.0 + torch.erf(h / math.sqrt(2.0)))
    return torch.clamp(h, min=0.0) ** 2  # relu2


def _ffn(x: torch.Tensor, ffn, use_swiglu: bool, activation: str = "silu") -> torch.Tensor:
    if use_swiglu:
        return (_act(x @ ffn.w1.weight.T, activation) * (x @ ffn.w3.weight.T)) @ ffn.w2.weight.T
    return _act(x @ ffn.fc1.weight.T, activation) @ ffn.fc2.weight.T


def reference_model(model) -> torch.nn.Module:
    """A float64 CPU deep copy whose *weights* the reference reads (differentiable leaves)."""
    m = copy.deepcopy(model).to("cpu").double().eval()
    for p in m.parameters():
        p.grad = None
    return m


def reference_forward(
    rm,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    window: int | None = None,
) -> torch.Tensor:
    """Logits ``(B,T,V)`` in float64 from the weights of ``rm`` (see :func:`reference_model`)."""
    c = rm.config
    ids = input_ids.cpu()
    B, T = ids.shape
    if position_ids is None:
        position_ids = torch.arange(T).expand(B, T)
    position_ids = position_ids.cpu()
    window = c.sliding_window if window is None else window
    key_padding = None if attention_mask is None else attention_mask.cpu()
    x = rm.embed_tokens.weight[ids]
    if not c.use_rope and not c.use_alibi:
        x = x + rm.embed_positions.embedding.weight[position_ids]
    for blk in rm.blocks:
        a = blk.attn
        h = _norm(x, blk.attn_norm, c.use_rmsnorm, c.norm_eps)
        x = x + ref_attention(
            h,
            a.q_proj.weight,
            a.k_proj.weight,
            a.v_proj.weight,
            a.o_proj.weight,
            c.num_heads,
            c.num_kv_heads,
            position_ids if c.use_rope else None,
            c.rope_base,
            key_padding,
            window,
            a.q_proj.bias,
            a.k_proj.bias,
            a.v_proj.bias,
            c.use_alibi,
        )
        x = x + _ffn(_norm(x, blk.ffn_norm, c.use_rmsnorm, c.norm_eps), blk.ffn, c.use_swiglu, c.ffn_act)
    return _norm(x, rm.final_norm, c.use_rmsnorm, c.norm_eps) @ rm.lm_head.weight.T


def reference_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token-mean next-token cross-entropy, ignoring label -100 (written out, no ``F.cross_entropy``)."""
    lg, lb = logits[:, :-1], labels.cpu()[:, 1:]
    logp = lg - torch.logsumexp(lg, dim=-1, keepdim=True)
    keep = lb != -100
    picked = logp.gather(-1, lb.clamp(min=0)[..., None]).squeeze(-1)
    return -(picked * keep).sum() / keep.sum()
