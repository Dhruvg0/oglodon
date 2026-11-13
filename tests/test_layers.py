import math

import pytest
import torch
import torch.nn.functional as F

from oglodon.layers import (
    GELUMLP,
    CausalSelfAttention,
    KVCache,
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    apply_rope,
    build_attention_bias,
)


def test_rmsnorm_matches_formula_and_unit_rms():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16) * 4 + 2
    norm = RMSNorm(16, eps=1e-6)
    y = norm(x)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(y, ref, atol=1e-6)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3, 5), atol=1e-4)
    # differs from LayerNorm (which removes the mean)
    assert not torch.allclose(y, F.layer_norm(x, (16,)), atol=1e-3)


def test_rmsnorm_learnable_gain():
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.fill_(3.0)
    x = torch.randn(2, 8)
    assert torch.allclose(norm(x), 3 * RMSNorm(8)(x), atol=1e-6)


def test_swiglu_matches_formula():
    torch.manual_seed(0)
    ffn = SwiGLU(8, 20)
    x = torch.randn(4, 8)
    ref = (F.silu(x @ ffn.w1.weight.T) * (x @ ffn.w3.weight.T)) @ ffn.w2.weight.T
    assert torch.allclose(ffn(x), ref, atol=1e-6)


def test_gelu_mlp_matches_formula():
    ffn = GELUMLP(8, 20)
    x = torch.randn(4, 8)
    assert torch.allclose(ffn(x), F.gelu(x @ ffn.fc1.weight.T) @ ffn.fc2.weight.T, atol=1e-6)


def _complex_rope(x, positions, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2).float() / d)
    z = torch.view_as_complex(x.reshape(*x.shape[:-1], d // 2, 2).contiguous())
    rot = torch.polar(torch.ones(len(positions), d // 2), positions[:, None].float() * theta[None])
    return torch.view_as_real(z * rot).flatten(-2)


def test_rope_matches_complex_rotation_reference():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 6, 8)
    cos, sin = RotaryEmbedding(8, 16)(torch.arange(6)[None])
    assert torch.allclose(
        apply_rope(x, cos, sin), _complex_rope(x[0, 0], torch.arange(6))[None, None], atol=1e-5
    )


def test_rope_position_zero_is_identity_and_norm_preserved():
    x = torch.randn(2, 3, 5, 8)
    cos, sin = RotaryEmbedding(8, 16)(torch.arange(5)[None].expand(2, 5))
    y = apply_rope(x, cos, sin)
    assert torch.allclose(y[:, :, 0], x[:, :, 0], atol=1e-6)
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_scores_depend_only_on_relative_offset():
    torch.manual_seed(0)
    q, k = torch.randn(8), torch.randn(8)
    rope = RotaryEmbedding(8, 64)

    def score(m, n):
        c, s = rope(torch.tensor([[m, n]]))
        qq = apply_rope(q.view(1, 1, 1, 8).expand(1, 1, 2, 8), c, s)[0, 0, 0]
        kk = apply_rope(k.view(1, 1, 1, 8).expand(1, 1, 2, 8), c, s)[0, 0, 1]
        return (qq @ kk).item()

    assert score(5, 2) == pytest.approx(score(25, 22), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 3), abs=1e-3)


def test_attention_bias_causal_with_cache_offset_and_padding():
    b = build_attention_bias(2, 5, torch.device("cpu"), torch.float32)  # 3 cached + 2 new
    allowed = b[0, 0] == 0
    assert allowed.tolist() == [[True, True, True, True, False], [True] * 5]
    pad = build_attention_bias(2, 2, torch.device("cpu"), torch.float32, torch.tensor([[0, 1]]))
    assert (pad[0, 0] == 0).tolist() == [[False, False], [False, True]]
    assert torch.isfinite(pad).all()  # no -inf, so a fully masked row cannot give NaN


def test_gqa_head_mapping_matches_explicit_loop():
    """Query head h must use K/V head h // group_size."""
    torch.manual_seed(0)
    H, KV, hd = 4, 2, 4
    attn = CausalSelfAttention(16, H, KV)
    x = torch.randn(1, 5, 16)
    out, w = attn(x, output_attentions=True)
    q = attn.q_proj(x).view(1, 5, H, hd)
    k = attn.k_proj(x).view(1, 5, KV, hd)
    v = attn.v_proj(x).view(1, 5, KV, hd)
    heads = []
    for h in range(H):
        kvh = h // (H // KV)
        s = (q[0, :, h] @ k[0, :, kvh].T) / math.sqrt(hd)
        s = s.masked_fill(torch.ones(5, 5).triu(1).bool(), float("-inf"))
        heads.append(s.softmax(-1) @ v[0, :, kvh])
    ref = attn.o_proj(torch.cat(heads, dim=-1))[None]
    assert torch.allclose(out, ref, atol=1e-5)
    # heads 0,1 share K/V head 0 and heads 2,3 share K/V head 1
    assert attn.group_size == 2


@pytest.mark.parametrize("kv", [1, 2, 4])
def test_kv_projection_sizes_shrink_with_fewer_kv_heads(kv):
    attn = CausalSelfAttention(32, 4, kv)
    assert attn.k_proj.out_features == kv * 8 == attn.v_proj.out_features


def test_attention_weights_rows_sum_to_one_and_are_causal():
    attn = CausalSelfAttention(16, 4, 2)
    _, w = attn(torch.randn(2, 6, 16), output_attentions=True)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 6), atol=1e-5)
    assert (w.triu(1) == 0).all()


def test_kv_cache_concatenates_and_counts_bytes():
    cache = KVCache(2)
    assert cache.seq_len == 0
    cache.update(0, torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))
    k, _ = cache.update(0, torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4))
    assert k.shape[2] == 4 and cache.seq_len == 4
    assert cache.nbytes() == 2 * 1 * 2 * 4 * 4 * 4
