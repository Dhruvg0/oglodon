import pytest
import torch
import torch.nn.functional as F

from oglodon import Oglodon, OglodonOutput

from .conftest import tiny_config


def ids(config, b=2, t=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, config.vocab_size, (b, t), generator=g)


def test_forward_shape_and_plain_tensor_return(variant_model):
    x = ids(variant_model.config)
    logits = variant_model(x)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 12, variant_model.config.vocab_size)


def test_rich_output_only_when_requested(model):
    x = ids(model.config)
    out = model(x, labels=x, use_cache=True, output_attentions=True)
    assert isinstance(out, OglodonOutput)
    assert out.past_key_values.seq_len == 12
    assert len(out.attentions) == model.config.num_layers
    assert model(x, use_cache=True).attentions is None


def test_loss_matches_manual_shifted_cross_entropy(model):
    x = ids(model.config)
    out = model(x, labels=x)
    ref = F.cross_entropy(out.logits[:, :-1].reshape(-1, model.config.vocab_size), x[:, 1:].reshape(-1))
    assert out.loss.item() == pytest.approx(ref.item(), abs=1e-6)
    # an untrained model is near uniform: loss ~ ln(V)
    assert abs(out.loss.item() - torch.log(torch.tensor(float(model.config.vocab_size))).item()) < 0.5


def test_loss_ignores_minus_100(model):
    x = ids(model.config)
    labels = x.clone()
    labels[:, :6] = -100
    full = model(x)
    ref = F.cross_entropy(full[:, 5:-1].reshape(-1, model.config.vocab_size), x[:, 6:].reshape(-1))
    assert model(x, labels=labels).loss.item() == pytest.approx(ref.item(), abs=1e-6)


def test_gradients_reach_every_parameter(variant_model):
    m = variant_model.train()
    x = ids(m.config)
    m(x, labels=x).loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_causal_masking_future_tokens_do_not_affect_past(variant_model):
    x = ids(variant_model.config, t=16)
    y = x.clone()
    y[:, 9:] = (y[:, 9:] + 7) % variant_model.config.vocab_size
    a, b = variant_model(x), variant_model(y)
    assert torch.allclose(a[:, :9], b[:, :9], atol=1e-6)
    assert not torch.allclose(a[:, 9:], b[:, 9:], atol=1e-3)


def test_kv_cache_matches_full_forward(variant_model):
    x = ids(variant_model.config, t=20)
    full = variant_model(x)
    out = variant_model(x[:, :8], use_cache=True)
    parts, cache = [out.logits], out.past_key_values
    for i in range(8, 20):
        o = variant_model(x[:, i : i + 1], past_key_values=cache, use_cache=True)
        parts.append(o.logits)
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-5)


def test_cache_chunked_prefill_matches_full(model):
    x = ids(model.config, t=18)
    o1 = model(x[:, :7], use_cache=True)
    o2 = model(x[:, 7:], past_key_values=o1.past_key_values, use_cache=True)
    assert torch.allclose(torch.cat([o1.logits, o2.logits], 1), model(x), atol=1e-5)


def test_left_padding_with_mask_matches_unpadded(variant_model):
    m, c = variant_model, variant_model.config
    x = ids(c, b=1, t=9)
    pad = torch.zeros(1, 4, dtype=torch.long)
    padded = torch.cat([pad, x], 1)
    mask = torch.cat([torch.zeros(1, 4), torch.ones(1, 9)], 1).long()
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    got = m(padded, attention_mask=mask, position_ids=pos)[:, 4:]
    assert torch.allclose(got, m(x), atol=1e-5)


def test_explicit_position_ids_matter_for_rope():
    m = Oglodon(tiny_config(init_std=0.3)).eval()  # large init so attention visibly affects logits
    x = ids(m.config, t=6)
    default = m(x)
    shifted = m(x, position_ids=torch.arange(6).expand(2, 6) + 5)
    # only relative positions matter for RoPE attention, so a uniform shift changes nothing...
    assert torch.allclose(default, shifted, atol=1e-4)
    # ...but a non-uniform change does
    other = m(x, position_ids=torch.tensor([[0, 1, 2, 3, 9, 10]]).expand(2, 6))
    assert not torch.allclose(default, other, atol=1e-3)
    assert (default - other).abs().max() > 1e-2


def test_input_validation(model):
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.zeros(1, 49, dtype=torch.long))
    with pytest.raises(ValueError, match="batch, seq"):
        model(torch.zeros(5, dtype=torch.long))
    with pytest.raises(ValueError, match="attention_mask"):
        model(torch.zeros(1, 4, dtype=torch.long), attention_mask=torch.ones(1, 3))


def test_weight_tying_and_param_count():
    tied, untied = Oglodon(tiny_config()), Oglodon(tiny_config(tie_embeddings=False))
    assert tied.lm_head.weight is tied.embed_tokens.weight
    assert untied.num_parameters() - tied.num_parameters() == 37 * 32


def test_dropout_only_active_in_train_mode():
    m = Oglodon(tiny_config(dropout=0.5))
    x = ids(m.config)
    m.eval()
    assert torch.equal(m(x), m(x))
    m.train()
    assert not torch.equal(m(x), m(x))


def test_runs_on_available_devices(device):
    m = Oglodon(tiny_config()).to(device).eval()
    x = ids(m.config).to(device)
    out = m.generate(x[:, :4], max_new_tokens=4)
    assert out.device.type == device and out.shape == (2, 8)
    cpu = Oglodon(tiny_config()).eval()
    cpu.load_state_dict({k: v.cpu() for k, v in m.state_dict().items()})
    assert torch.allclose(cpu(x.cpu()), m(x).cpu(), atol=1e-4)
