"""SDPA causal fast path, gradient checkpointing, Trainer options, torch.compile."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from oglodon import Oglodon, OglodonConfig
from oglodon.training import Trainer

CFG = dict(vocab_size=40, hidden_size=32, num_layers=3, num_heads=4, num_kv_heads=2, max_seq_len=48)


def make(**kw):
    torch.manual_seed(0)
    return Oglodon(OglodonConfig(**{**CFG, **kw}))


def test_sdpa_fast_path_is_used_only_when_no_extra_mask_is_needed(monkeypatch):
    calls = []
    real = F.scaled_dot_product_attention

    def spy(*a, **k):
        calls.append({"mask": k.get("attn_mask") is not None, "causal": k.get("is_causal", False)})
        return real(*a, **k)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    m = make(attn_impl="sdpa").eval()
    x = torch.randint(0, 40, (2, 10))
    m(x)
    assert calls and all(c["causal"] and not c["mask"] for c in calls)  # plain causal -> flash-eligible
    calls.clear()
    m(x, attention_mask=torch.ones(2, 10, dtype=torch.long))
    assert all(c["mask"] and not c["causal"] for c in calls)  # padding mask given -> explicit mask
    calls.clear()
    make(attn_impl="sdpa", sliding_window=4).eval()(x)
    assert all(c["mask"] for c in calls)  # local window needs a mask
    calls.clear()
    out = m(x[:, :6], use_cache=True)
    calls.clear()
    m(x[:, 6:7], past_key_values=out.past_key_values, use_cache=True)  # one new token: no mask at all
    assert calls and all(not c["mask"] and not c["causal"] for c in calls)
    calls.clear()
    m(
        x[:, 6:9], past_key_values=out.past_key_values, use_cache=True
    )  # chunk after a cache: needs the offset mask
    assert all(c["mask"] for c in calls)


@pytest.mark.parametrize("kw", [{}, {"use_rope": False}, {"use_swiglu": False, "qkv_bias": True}])
def test_fast_path_matches_eager_including_cache_steps(kw):
    eager, fast = make(**kw).eval(), make(attn_impl="sdpa", **kw).eval()
    fast.load_state_dict(eager.state_dict())
    x = torch.randint(0, 40, (3, 12))
    assert torch.allclose(eager(x), fast(x), atol=1e-5)
    a, b = eager(x[:, :5], use_cache=True), fast(x[:, :5], use_cache=True)
    for t in range(5, 12):
        a = eager(x[:, t : t + 1], past_key_values=a.past_key_values, use_cache=True)
        b = fast(x[:, t : t + 1], past_key_values=b.past_key_values, use_cache=True)
        assert torch.allclose(a.logits, b.logits, atol=1e-5)
    assert torch.allclose(a.logits[:, -1], eager(x)[:, -1], atol=1e-5)


def _grads(model, x, seed=3):
    model.zero_grad()
    torch.manual_seed(seed)
    loss = model(x, labels=x).loss
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_gradient_checkpointing_gives_identical_loss_and_gradients(dropout):
    x = torch.randint(0, 40, (4, 16))
    plain = make(dropout=dropout).train()
    ckpt = copy.deepcopy(plain)
    ckpt.gradient_checkpointing_enable()
    l1, g1 = _grads(plain, x)
    l2, g2 = _grads(ckpt, x)
    assert l1 == pytest.approx(l2, rel=1e-6)
    assert g1.keys() == g2.keys()
    for n in g1:  # dropout masks are replayed in the recomputation, so even dropout runs agree
        assert torch.allclose(g1[n], g2[n], atol=1e-6, rtol=1e-5), n


def test_gradient_checkpointing_stores_fewer_activations():
    x = torch.randint(0, 40, (4, 32))

    def saved_bytes(model):
        total = 0

        def pack(t):
            nonlocal total
            total += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            _ = model(x, labels=x).loss
        return total

    plain = make(num_layers=6).train()
    ckpt = copy.deepcopy(plain)
    ckpt.gradient_checkpointing_enable()
    assert saved_bytes(ckpt) < 0.6 * saved_bytes(plain)


def test_gradient_checkpointing_is_inert_in_eval_cache_and_attention_output():
    m = make().eval()
    m.gradient_checkpointing_enable()
    x = torch.randint(0, 40, (2, 8))
    ref = make().eval()(x)
    assert torch.equal(m(x), ref)
    out = m(x, use_cache=True, output_attentions=True)
    assert out.past_key_values is not None and len(out.attentions) == 3
    m.train()
    out = m(x[:, :4], use_cache=True)  # training mode + cache: must not checkpoint (cache would be corrupted)
    assert out.past_key_values.seq_len == 4
    m.gradient_checkpointing_disable()
    assert not m.gradient_checkpointing


def _train(dropout=0.1, **trainer_kw):
    model = make(dropout=dropout)
    data = [torch.randint(0, 40, (4, 16), generator=torch.Generator().manual_seed(i)) for i in range(6)]
    loader = DataLoader([(d[i], d[i]) for d in data for i in range(4)], batch_size=4, shuffle=False)
    tr = Trainer(model, loader, max_steps=6, learning_rate=1e-2, warmup_steps=0, log_interval=0, device="cpu",
                 logger=None, **trainer_kw)  # fmt: skip
    torch.manual_seed(7)
    tr.train()
    return model


def test_trainer_gradient_checkpointing_trains_to_the_same_weights():
    a, b = _train(), _train(gradient_checkpointing=True)
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.allclose(p, q, atol=1e-5, rtol=1e-4), n


@pytest.mark.slow
def test_trainer_with_torch_compile_matches_eager():
    pytest.importorskip("torch._dynamo")
    # dropout 0: a compiled graph draws its dropout masks from a different RNG stream than eager mode
    a, b = _train(dropout=0.0), _train(dropout=0.0, compile=True)
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.allclose(p, q, atol=1e-4, rtol=1e-3), n
    assert not any(k.startswith("_orig_mod") for k in b.state_dict())  # checkpoints keep clean key names
