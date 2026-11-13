"""Regression tests for the bugs found in the audit (each failed before its fix)."""

import pytest
import torch
from torch.utils.data import DataLoader

from oglodon import Oglodon
from oglodon.generation import generate
from oglodon.layers import KVCache
from oglodon.layers.attention import build_attention_bias
from oglodon.training import Trainer
from oglodon.validation import reference as ref

from .conftest import tiny_config


def loader(n=16, T=10, seed=0, ragged=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(2, 37, (n, T), generator=g)
    y = x.clone()
    if ragged:
        for i in range(n):
            y[i, : (i * 3) % T] = -100
    return DataLoader(list(zip(x, y, strict=True)), batch_size=2, shuffle=False)


def make_trainer(**kw):
    torch.manual_seed(0)
    return Trainer(Oglodon(tiny_config(**kw.pop("cfg", {}))), loader(), logger=None, device="cpu", **kw)


def test_generate_eos_none_disables_eos_but_unset_uses_config():
    torch.manual_seed(0)
    m = Oglodon(tiny_config(eos_token_id=5, tie_embeddings=False))
    with torch.no_grad():
        m.lm_head.weight.zero_()
        m.lm_head.weight[5] = 1.0
        m.embed_tokens.weight[:] = 1.0  # hidden ~ constant positive -> logit 5 = sum(h) > 0 = others
    x = torch.tensor([[2, 3]])
    assert generate(m, x, 6).shape[1] == 3  # stops at EOS after one token
    assert generate(m, x, 6, eos_token_id=None).shape[1] == 8  # explicit None: no EOS handling


def test_position_ids_are_validated():
    m = Oglodon(tiny_config()).eval()
    x = torch.randint(0, 37, (2, 5))
    with pytest.raises(ValueError):
        m(x, position_ids=torch.zeros(2, 4, dtype=torch.long))
    with pytest.raises(ValueError):
        m(x, position_ids=torch.full((2, 5), 48))
    with pytest.raises(ValueError):
        m(x, position_ids=torch.full((2, 5), -1))
    m(x, position_ids=torch.arange(5).expand(2, 5))


def test_kv_cache_reset_and_batch_mismatch():
    m = Oglodon(tiny_config()).eval()
    out = m(torch.randint(0, 37, (2, 4)), use_cache=True)
    cache = out.past_key_values
    assert cache.seq_len == 4
    cache.reset()
    assert cache.seq_len == 0
    m(torch.randint(0, 37, (2, 3)), past_key_values=cache, use_cache=True)
    assert cache.seq_len == 3
    with pytest.raises(ValueError):
        m(torch.randint(0, 37, (3, 1)), past_key_values=cache, use_cache=True)
    assert isinstance(cache, KVCache)


@pytest.mark.parametrize("window", [None, 1, 3])
@pytest.mark.parametrize("q_len,kv_len", [(6, 6), (1, 6), (3, 6)])
def test_attention_bias_matches_independent_mask(window, q_len, kv_len):
    pad = torch.ones(2, kv_len, dtype=torch.long)
    pad[1, :2] = 0
    bias = build_attention_bias(q_len, kv_len, torch.device("cpu"), torch.float32, pad, window)
    want = ref.ref_mask(q_len, kv_len, pad, window)
    got = bias[:, 0] == 0
    assert torch.equal(got, want.expand_as(got) if want.shape != got.shape else want)


@pytest.mark.parametrize("arch", [{}, {"num_kv_heads": 1}, {"sliding_window": 3}])
def test_eager_equals_sdpa_forward(arch):
    torch.manual_seed(0)
    m = Oglodon(tiny_config(init_std=0.3, **arch)).eval()
    x = torch.randint(0, 37, (2, 9))
    mask = torch.ones(2, 9, dtype=torch.long)
    mask[1, :3] = 0
    a = m(x, attention_mask=mask)
    m.set_attn_impl("sdpa")
    b = m(x, attention_mask=mask)
    keep = mask.bool()  # fully masked pad rows are unspecified; compare real rows
    assert (a[keep] - b[keep]).abs().max() < 1e-5


def test_bf16_model_is_finite_and_close_to_float32():
    torch.manual_seed(0)
    m = Oglodon(tiny_config(init_std=0.1)).eval()
    x = torch.randint(0, 37, (2, 12))
    full = m(x)
    half = Oglodon(tiny_config(init_std=0.1)).eval()
    half.load_state_dict(m.state_dict())
    out = half.to(torch.bfloat16)(x)
    assert torch.isfinite(out).all() and (out.float() - full).abs().max() < 0.15


def test_float64_model_is_not_silently_downcast():
    """Softmax, RMSNorm, RoPE tables and the loss used to round float64 models to float32."""
    from oglodon.validation import validate_model

    r = validate_model(Oglodon(tiny_config()), dtype="float64", select=["model.reference", "kv_cache.logits"])
    assert r.passed and all(c.max_abs_diff < 1e-8 for c in r.checks), str(r)


def test_compute_gradients_equals_one_big_batch():
    t = make_trainer()
    batches = [{"input_ids": torch.randint(2, 37, (2, 8))} for _ in range(3)]
    for i, b in enumerate(batches):
        b["labels"] = b["input_ids"].clone()
        b["labels"][:, : i * 2] = -100  # ragged supervision
    t.compute_gradients(batches)
    got = [p.grad.clone() for p in t.model.parameters()]
    big = {k: torch.cat([b[k] for b in batches]) for k in ("input_ids", "labels")}
    t.optimizer.zero_grad()
    t.model(**big).loss.backward()
    for g, p in zip(got, t.model.parameters(), strict=True):
        assert torch.allclose(g, p.grad, atol=1e-6)


def test_compute_gradients_skips_zero_token_microbatch_and_all_empty_returns_nan():
    t = make_trainer()
    x = torch.randint(2, 37, (2, 6))
    empty = {"input_ids": x, "labels": torch.full_like(x, -100)}
    full = {"input_ids": x, "labels": x.clone()}
    loss = t.compute_gradients([empty, full])
    assert torch.isfinite(torch.tensor(loss)) and all(
        torch.isfinite(p.grad).all() for p in t.model.parameters()
    )
    assert torch.isnan(torch.tensor(t.compute_gradients([empty, empty])))


def test_evaluate_is_token_weighted_and_survives_empty_batches():
    torch.manual_seed(0)
    m = Oglodon(tiny_config())
    dl = loader(ragged=True)
    t = Trainer(m, loader(), eval_dataloader=dl, logger=None, device="cpu")
    got = t.evaluate()
    m.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y in dl:
            k = int((y[:, 1:] != -100).sum())
            if k:
                tot += m(x, labels=y).loss.item() * k
                n += k
    assert abs(got - tot / n) < 1e-5


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_resume_is_exact(tmp_path, dropout):
    kw = dict(max_steps=8, learning_rate=1e-2, grad_accum_steps=2, warmup_steps=2, cfg={"dropout": dropout})
    a = make_trainer(**{**kw, "cfg": dict(kw["cfg"])})
    a.train()
    b = make_trainer(**{**kw, "cfg": dict(kw["cfg"])})
    b.train(until=4)
    b.save_checkpoint(tmp_path / "ck")
    torch.manual_seed(123)  # a different RNG state must not matter after resume
    c = make_trainer(**{**kw, "cfg": dict(kw["cfg"])})
    c.load_checkpoint(tmp_path / "ck")
    c.train()
    assert c.step == 8
    for pa, pc in zip(a.model.parameters(), c.model.parameters(), strict=True):
        assert torch.equal(pa, pc)


def test_train_until_does_not_change_schedule():
    a = make_trainer(max_steps=6, warmup_steps=1)
    a.train()
    b = make_trainer(max_steps=6, warmup_steps=1)
    b.train(until=3)
    b.train()
    assert a.scheduler.get_last_lr() == b.scheduler.get_last_lr()
