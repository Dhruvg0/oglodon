"""Streaming generation, stop criteria, repetition penalty, min-p."""

import pytest
import torch

from oglodon import Oglodon, OglodonConfig
from oglodon.generation import apply_repetition_penalty, filter_logits, generate, generate_stream, iter_text
from oglodon.tokenization import BPETokenizer


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = OglodonConfig(vocab_size=40, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                        max_seq_len=64, pad_token_id=0, eos_token_id=1)  # fmt: skip
    return Oglodon(cfg).eval()


def prompts(n=3, t=6, seed=1):
    return torch.randint(2, 40, (n, t), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("use_cache", [True, False])
@pytest.mark.parametrize("temperature", [0.0, 0.9])
def test_stream_equals_generate(model, use_cache, temperature):
    x = prompts()
    kw = dict(max_new_tokens=12, temperature=temperature, top_k=10, use_cache=use_cache, eos_token_id=None)
    full = generate(model, x, generator=torch.Generator().manual_seed(5), **kw)
    toks = list(generate_stream(model, x, generator=torch.Generator().manual_seed(5), **kw))
    assert len(toks) == 12 and all(t.shape == (3,) for t in toks)
    assert torch.equal(torch.cat([x, torch.stack(toks, 1)], 1), full)


def test_stream_validates_eagerly_and_restores_training_mode(model):
    with pytest.raises(ValueError, match="max_seq_len"):
        generate_stream(model, prompts(), max_new_tokens=500)  # raised now, not on first next()
    with pytest.raises(ValueError, match="outside the vocabulary"):
        generate_stream(model, prompts(), stop_token_ids=[99])
    with pytest.raises(ValueError, match="empty"):
        generate_stream(model, prompts(), stop_sequences=[[]])
    model.train()
    try:
        s = generate_stream(model, prompts(), max_new_tokens=5)
        next(s)
        assert not model.training  # eval while streaming
        s.close()  # abandoning a stream restores the mode
        assert model.training
    finally:
        model.eval()


def test_stop_token_ids_finish_rows_and_pad_afterwards(model):
    x = prompts()
    base = generate(model, x, max_new_tokens=10, eos_token_id=None)[:, 6:]
    stop = int(base[0, 3])
    first = [(row == stop).nonzero()[0, 0].item() if (row == stop).any() else None for row in base]
    out = generate(model, x, max_new_tokens=10, eos_token_id=None, stop_token_ids=[stop], pad_token_id=0)[
        :, 6:
    ]
    for r, f in enumerate(first):
        if f is None:
            assert torch.equal(out[r], base[r][: out.shape[1]])
        else:
            assert torch.equal(out[r, : f + 1], base[r, : f + 1])  # up to and including the stop token
            assert (out[r, f + 1 :] == 0).all()  # padded afterwards
    f0 = first[0]
    assert f0 is not None and out.shape[1] <= 10


def test_generation_ends_early_when_all_rows_stop(model):
    x = prompts(n=1)
    base = generate(model, x, max_new_tokens=10, eos_token_id=None)[0, 6:].tolist()
    stop = base[2]
    first = base.index(stop)  # first occurrence (<= 2)
    out = generate(model, x, max_new_tokens=10, eos_token_id=None, stop_token_ids=[stop])
    assert out.shape[1] == 6 + first + 1 and out[0, -1].item() == stop
    assert out[0, 6:].tolist() == base[: first + 1]


def test_stop_sequences(model):
    x = prompts(n=2)
    base = generate(model, x, max_new_tokens=12, eos_token_id=None)[:, 6:]
    seq = base[1, 4:7].tolist()
    out = generate(model, x, max_new_tokens=12, eos_token_id=None, stop_sequences=[seq], pad_token_id=0)[
        :, 6:
    ]

    def end_of_first_match(row):
        r = row.tolist()
        for i in range(len(r) - 2):
            if r[i : i + 3] == seq:
                return i + 3
        return None

    for b in range(2):
        e = end_of_first_match(base[b])
        if e is None:
            assert torch.equal(out[b], base[b][: out.shape[1]])
        else:
            assert torch.equal(out[b, :e], base[b, :e]) and (out[b, e:] == 0).all()
    assert (
        end_of_first_match(base[1]) is not None and end_of_first_match(base[1]) <= 7
    )  # the seed sequence occurs
    assert out.shape[1] >= end_of_first_match(base[1])


def test_default_behaviour_is_unchanged_when_new_options_are_off(model):
    x = prompts()
    a = generate(model, x, max_new_tokens=9)
    b = generate(model, x, max_new_tokens=9, repetition_penalty=1.0, stop_token_ids=None, stop_sequences=None)
    assert torch.equal(a, b)


def test_repetition_penalty_matches_manual_formula_and_ignores_padding():
    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0, -2.0]])
    ids = torch.tensor([[0, 1, 1, 4]])
    mask = torch.tensor([[1, 1, 1, 0]])  # id 4 is padding: must not be penalised
    got = apply_repetition_penalty(logits, ids, mask, 2.0)
    assert torch.allclose(got, torch.tensor([[1.0, -2.0, 0.5, 3.0, -2.0]]))
    assert torch.equal(apply_repetition_penalty(logits, ids, mask, 1.0), logits)
    # a padding id equal to a real token elsewhere must still be penalised (scatter conflict regression)
    ids2, mask2 = torch.tensor([[1, 1]]), torch.tensor([[1, 0]])
    assert apply_repetition_penalty(logits, ids2, mask2, 2.0)[0, 1] == -2.0
    with pytest.raises(ValueError):
        apply_repetition_penalty(logits, ids, mask, 0.0)


def test_repetition_penalty_reduces_repeats(model):
    x = prompts(n=4)
    plain = generate(model, x, max_new_tokens=30, eos_token_id=None)[:, 6:]
    pen = generate(model, x, max_new_tokens=30, eos_token_id=None, repetition_penalty=3.0)[:, 6:]

    def distinct(t):
        return sum(len(set(r.tolist())) for r in t)

    assert distinct(pen) > distinct(plain)


def test_streaming_with_left_padding_matches_unpadded(model):
    a, b = prompts(1, 4, 3), prompts(1, 7, 4)
    pad = torch.zeros(1, 3, dtype=torch.long)
    batch = torch.cat([torch.cat([pad, a], 1), b], 0)
    mask = torch.cat([torch.cat([torch.zeros(1, 3), torch.ones(1, 4)], 1), torch.ones(1, 7)], 0).long()
    toks = torch.stack(
        list(generate_stream(model, batch, max_new_tokens=8, attention_mask=mask, eos_token_id=None)), 1
    )
    for r, single in enumerate((a, b)):
        want = generate(model, single, max_new_tokens=8, eos_token_id=None)[:, single.shape[1] :]
        assert torch.equal(toks[r : r + 1], want)


def test_min_p_filter():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    kept = filter_logits(logits, min_p=0.5).isfinite()  # keep p >= 0.5 * 0.5 = 0.25
    assert kept.tolist() == [[True, True, False, False]]
    assert filter_logits(logits, min_p=0.0).isfinite().all()
    assert filter_logits(logits, min_p=1.0).isfinite().sum() == 1
    with pytest.raises(ValueError):
        filter_logits(logits, min_p=1.5)


def test_min_p_sampling_never_picks_filtered_tokens():
    logits = torch.log(torch.tensor([[0.6, 0.35, 0.04, 0.01]])).repeat(500, 1)
    from oglodon.generation import sample_next_token

    picks = sample_next_token(logits, 1.0, min_p=0.2, generator=torch.Generator().manual_seed(0))
    assert set(picks.tolist()) <= {0, 1} and {0, 1} <= set(picks.tolist())


def test_iter_text_reassembles_exactly_with_multibyte_text(model):
    tok = BPETokenizer.train("héllo wörld 日本語 ñandú " * 30, 300)
    cfg = OglodonConfig(vocab_size=tok.vocab_size, hidden_size=32, num_layers=1, num_heads=4, max_seq_len=64)
    m = Oglodon(cfg).eval()
    torch.manual_seed(0)
    x = torch.tensor([tok.encode("héllo")])
    gen = torch.Generator().manual_seed(2)
    pieces = list(iter_text(tok, m.generate_stream(x, max_new_tokens=25, temperature=1.0, generator=gen)))
    gen = torch.Generator().manual_seed(2)
    full = m.generate(x, max_new_tokens=25, temperature=1.0, generator=gen)[0, x.shape[1] :].tolist()
    assert "".join(pieces) == tok.decode(full)

    # valid text split across byte-level tokens: no piece may expose a half character
    text = "日本語 héllo ñandú"
    byte_tok = BPETokenizer.train("abc", 260)  # (almost) no merges: tokens are raw bytes
    ids = byte_tok.encode(text)
    assert len(ids) > len(text)  # multibyte characters really are split over several tokens
    steps = (torch.tensor([i]) for i in ids)
    out = list(iter_text(byte_tok, steps))
    assert "".join(out) == text and all("\ufffd" not in piece for piece in out)
