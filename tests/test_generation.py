import pytest
import torch

from oglodon.generation import filter_logits, sample_next_token


def prompt(c, b=3, t=5):
    return torch.randint(2, c.vocab_size, (b, t), generator=torch.Generator().manual_seed(1))


def test_greedy_is_deterministic_and_cached_equals_uncached(variant_model):
    p = prompt(variant_model.config)
    a = variant_model.generate(p, max_new_tokens=12, use_cache=True)
    b = variant_model.generate(p, max_new_tokens=12, use_cache=False)
    assert torch.equal(a, b) and torch.equal(a, variant_model.generate(p, max_new_tokens=12))
    assert a.shape == (3, 17) and torch.equal(a[:, :5], p)


def test_sampling_reproducible_with_seeded_generator(model):
    p = prompt(model.config)
    kw = dict(max_new_tokens=10, temperature=0.9, top_k=10, top_p=0.9)
    a = model.generate(p, generator=torch.Generator().manual_seed(7), **kw)
    b = model.generate(p, generator=torch.Generator().manual_seed(7), **kw)
    c = model.generate(p, generator=torch.Generator().manual_seed(8), **kw)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_sampling_cached_equals_uncached_with_same_seed(model):
    p = prompt(model.config)
    kw = dict(max_new_tokens=10, temperature=1.0)
    a = model.generate(p, generator=torch.Generator().manual_seed(3), use_cache=True, **kw)
    b = model.generate(p, generator=torch.Generator().manual_seed(3), use_cache=False, **kw)
    assert torch.equal(a, b)


def test_generation_stops_early_when_every_row_hits_eos(model, monkeypatch):
    monkeypatch.setattr(
        "oglodon.generation.sampler.sample_next_token",
        lambda logits, *a, **k: torch.full((logits.shape[0],), 1),
    )
    out = model.generate(prompt(model.config), max_new_tokens=9)
    assert out.shape == (3, 6) and (out[:, -1] == 1).all()  # prompt 5 + a single eos


def test_finished_rows_are_padded_while_others_continue(model, monkeypatch):
    # Row 0 emits eos immediately, row 1 never does.
    calls = {"n": 0}

    def fake_sample(logits, *a, **k):
        calls["n"] += 1
        return torch.tensor([1, 9]) if calls["n"] == 1 else torch.tensor([7, 9])

    monkeypatch.setattr("oglodon.generation.sampler.sample_next_token", fake_sample)
    out = model.generate(prompt(model.config, b=2), max_new_tokens=4)
    gen = out[:, 5:]
    assert gen[0].tolist() == [1, 0, 0, 0]  # eos then pad (config pad id = 0)
    assert gen[1].tolist() == [9, 9, 9, 9]


def test_left_padded_batch_matches_individual_generation(variant_model):
    m, c = variant_model, variant_model.config
    short = torch.randint(2, c.vocab_size, (1, 3), generator=torch.Generator().manual_seed(5))
    long = torch.randint(2, c.vocab_size, (1, 7), generator=torch.Generator().manual_seed(6))
    batch = torch.cat([torch.cat([torch.zeros(1, 4, dtype=torch.long), short], 1), long], 0)
    mask = torch.tensor([[0] * 4 + [1] * 3, [1] * 7])
    for cache in (True, False):
        out = m.generate(batch, attention_mask=mask, max_new_tokens=6, eos_token_id=None, use_cache=cache)
        assert torch.equal(out[0, 7:], m.generate(short, max_new_tokens=6, eos_token_id=None)[0, 3:])
        assert torch.equal(out[1, 7:], m.generate(long, max_new_tokens=6, eos_token_id=None)[0, 7:])


def test_length_limit_and_argument_errors(model):
    p = prompt(model.config, t=10)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.generate(p, max_new_tokens=40)
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(torch.zeros(1, 4), temperature=-1)
    assert model.generate(p, max_new_tokens=0).shape == p.shape


def test_generate_restores_training_mode(model):
    model.train()
    model.generate(prompt(model.config), max_new_tokens=2)
    assert model.training


def test_top_k_keeps_exactly_k_tokens():
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0, 0.0]])
    out = filter_logits(logits, top_k=2)
    assert torch.isfinite(out).tolist() == [[False, True, True, False, False]]


def test_top_p_keeps_smallest_nucleus():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    assert torch.isfinite(filter_logits(logits, top_p=0.7)).tolist() == [[True, True, False, False]]
    assert torch.isfinite(filter_logits(logits, top_p=0.5)).tolist() == [[True, False, False, False]]
    assert torch.isfinite(filter_logits(logits, top_p=1.0)).all()


def test_sampling_never_picks_filtered_tokens():
    logits = torch.tensor([[0.0, 5.0, 4.9, -3.0, -4.0]]).expand(500, 5)
    g = torch.Generator().manual_seed(0)
    picks = sample_next_token(logits, temperature=1.0, top_k=2, generator=g)
    assert set(picks.tolist()) <= {1, 2} and len(set(picks.tolist())) == 2


def test_temperature_changes_sharpness():
    logits = torch.tensor([[2.0, 1.0, 0.0]]).expand(2000, 3)
    g = torch.Generator().manual_seed(0)
    hot = sample_next_token(logits, temperature=5.0, generator=g)
    cold = sample_next_token(logits, temperature=0.1, generator=torch.Generator().manual_seed(0))
    assert (cold == 0).float().mean() > 0.99 and (hot == 0).float().mean() < 0.6


def test_sampling_with_a_generator_on_another_device_is_handled():
    import torch

    from oglodon.generation.sampler import sample_next_token as sample_next

    logits = torch.randn(3, 10)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    assert torch.equal(sample_next(logits, 1.0, generator=g1), sample_next(logits, 1.0, generator=g2))
    if torch.cuda.is_available():  # the Colab bug: CPU generator, CUDA logits
        out = sample_next(logits.cuda(), 1.0, generator=torch.Generator().manual_seed(1))
        assert out.device.type == "cuda"
        assert torch.equal(out.cpu(), sample_next(logits, 1.0, generator=torch.Generator().manual_seed(1)))
